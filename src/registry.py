"""Реестр моделей и гейт качества (АР-5).

Задание требует «хранилище версий моделей и контроль качества»
(5.a.ii). До появления этого модуля на диске лежали три файла —
`dt_latest.pkl`, `mlp_latest.pkl`, `best_model.pkl` — без единого
метаданного: непонятно, какой версии модель, на каких данных и с каким
результатом она обучена, и по какому правилу выбрана лучшая.

Здесь три уровня:

1. **Неизменяемые артефакты** `models/<model>__v<NNN>__<батч>.pkl` —
   каждая версия сохраняется отдельно и никогда не перезаписывается.
2. **Указатели** `best_model.pkl` и `<model>_latest.pkl` — копии текущей
   продуктовой и последней обученной версии соответственно. Инференс
   работает с ними, не зная ничего о версионировании.
3. **`registry.json`** — реестр: по версии видно метрики, хэш
   конфигурации, отпечаток схемы признаков, результат проверки гейта
   и решение о продвижении.

**Гейт качества** решает, достоин ли кандидат стать продуктовой моделью.
Без него «контроль качества» означал бы только `argmax` по f1, что
продвигало бы заведомо худшую модель, если она случайно оказалась
лучше на одном батче. Решения гейта записываются в реестр, поэтому
видно не только «какая модель выбрана», но и «почему».
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils import atomic_pickle_dump, to_json_serializable

logger = logging.getLogger(__name__)

REGISTRY_FILE = "registry.json"
SCHEMA_VERSION = 1
_VERSION_PATTERN = re.compile(r"__(v\d+)__")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class GateResult:
    """Результат проверки кандидата гейтом качества."""

    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": self.checks, "reasons": self.reasons}


class QualityGate:
    """Правила допуска модели в продуктив.

    Набор правил задаётся конфигурацией, а не зашит в код, чтобы порог
    можно было менять, не переписывая конвейер. При отсутствии записи
    правило не применяется — это позволяет включить гейт частично.
    """

    def __init__(self, rules: dict[str, Any] | None = None) -> None:
        self.rules = rules or {}

    def evaluate(
        self,
        metrics: dict[str, Any],
        production: dict[str, Any] | None,
        context: dict[str, Any] | None = None,
    ) -> GateResult:
        """Проверить кандидата.

        Args:
            metrics: метрики кандидата на валидации.
            production: метрики текущей продуктовой версии, либо `None`.
            context: дополнительные сведения — например, значение PSI.

        Returns:
            `GateResult` с перечнем проверок и причин отказа.
        """
        context = context or {}
        checks: list[dict[str, Any]] = []
        reasons: list[str] = []

        min_f1 = self.rules.get("min_f1")
        if min_f1 is not None:
            value = float(metrics.get("f1") or 0.0)
            ok = value >= float(min_f1)
            checks.append(
                {"name": "min_f1", "threshold": min_f1, "value": value, "passed": ok}
            )
            if not ok:
                reasons.append(
                    f"f1 = {value:.4f} ниже порога {min_f1}"
                )

        min_roc_auc = self.rules.get("min_roc_auc")
        if min_roc_auc is not None:
            value = metrics.get("roc_auc")
            value = None if value is None else float(value)
            ok = value is not None and value >= float(min_roc_auc)
            checks.append(
                {"name": "min_roc_auc", "threshold": min_roc_auc,
                 "value": value, "passed": ok}
            )
            if not ok:
                reasons.append(
                    f"roc_auc = {value if value is not None else 'н/д'} "
                    f"ниже порога {min_roc_auc}"
                )

        max_regression = self.rules.get("max_regression_f1")
        if max_regression is not None and production is not None:
            candidate = float(metrics.get("f1") or 0.0)
            current = float(production.get("f1") or 0.0)
            drop = current - candidate
            ok = drop <= float(max_regression)
            checks.append(
                {"name": "max_regression_f1", "threshold": max_regression,
                 "value": round(drop, 6), "reference": current, "passed": ok}
            )
            if not ok:
                reasons.append(
                    f"f1 упал на {drop:.4f} относительно продуктовой версии "
                    f"({current:.4f} → {candidate:.4f}), допустимо {max_regression}"
                )

        max_psi = self.rules.get("max_psi")
        if max_psi is not None:
            psi = context.get("psi")
            value = None if psi is None else float(psi)
            if value is None:
                # Проверка не выполнялась (например, на первом батче ещё
                # нет эталона дрейфа), и это НЕ повод отклонить модель.
                # Считать отсутствие данных провалом было бы ошибкой:
                # гейт отклонил бы первую же модель в прогоне.
                checks.append(
                    {"name": "max_psi", "threshold": max_psi, "value": None,
                     "passed": True, "skipped": True,
                     "note": "дрейф ещё не рассчитан"}
                )
            else:
                ok = value <= float(max_psi)
                checks.append(
                    {"name": "max_psi", "threshold": max_psi, "value": value,
                     "passed": ok}
                )
                if not ok:
                    reasons.append(f"PSI = {value} превышает {max_psi}")

        return GateResult(passed=not reasons, checks=checks, reasons=reasons)


class ModelRegistry:
    """Файловое хранилище версий моделей."""

    def __init__(self, models_dir: str | os.PathLike[str], gate: QualityGate | None = None) -> None:
        self.models_dir = Path(models_dir)
        self.gate = gate or QualityGate()
        # Правила хранятся в самом гейте: реестр обращается к ним при
        # решении о продвижении, и дублировать конфигурацию в двух местах
        # означало бы риск расхождения.
        self.rules = self.gate.rules
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._state = self._read()

    # -- хранение состояния --------------------------------------------

    @property
    def path(self) -> Path:
        return self.models_dir / REGISTRY_FILE

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "production": None,
            "latest": {},
            "versions": [],
        }

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self._empty()
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError) as error:
            logger.warning("Реестр %s не читается (%s) — начинаем заново", self.path, error)
            return self._empty()
        if data.get("schema_version") != SCHEMA_VERSION:
            logger.warning("Схема реестра не совпадает — начинаем заново")
            return self._empty()
        return data

    def save(self) -> None:
        self._state["updated_at"] = _utc_now()
        temporary = self.path.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(to_json_serializable(self._state), handle, ensure_ascii=False, indent=2)
        os.replace(temporary, self.path)

    # -- доступ ---------------------------------------------------------

    @property
    def versions(self) -> list[dict[str, Any]]:
        return list(self._state.get("versions", []))

    @property
    def production(self) -> dict[str, Any] | None:
        return self._state.get("production")

    def production_metrics(self) -> dict[str, Any] | None:
        """Метрики текущей продуктовой версии — база для проверки регрессии."""
        production = self.production
        return production.get("metrics") if production else None

    def by_status(self, status: str) -> list[dict[str, Any]]:
        return [entry for entry in self.versions if entry.get("status") == status]

    def next_version_number(self) -> int:
        """Следующий порядковый номер версии по всему реестру."""
        highest = 0
        for entry in self.versions:
            match = _VERSION_PATTERN.search(entry.get("artifact", ""))
            if match:
                highest = max(highest, int(match.group(1)[1:]))
            else:
                highest = max(highest, int(str(entry.get("version", "v0"))[1:]))
        return highest + 1

    def artifact_name(self, model: str, version: int, batch_label: str) -> str:
        """Имя неизменяемого артефакта по схеме АР-16."""
        return f"{model}__v{version:04d}__{batch_label}.pkl"

    # -- регистрация ----------------------------------------------------

    def register(
        self,
        model: Any,
        model_name: str,
        batch_idx: int,
        batch_label: str,
        metrics: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Сохранить новую версию модели и проверить её гейтом.

        Каждая версия сохраняется всегда, независимо от результата
        гейта: регресс должен оставаться в истории, иначе невозможно
        понять, почему качество упало.

        Args:
            model: обученный объект модели.
            model_name: имя модели (`dt`, `mlp`, `lr`, …).
            batch_idx: индекс батча.
            batch_label: человекочитаемое имя батча.
            metrics: метрики на валидации.
            context: дополнительные сведения для гейта (например, PSI).

        Returns:
            Запись реестра для этой версии.
        """
        version = self.next_version_number()
        artifact = self.artifact_name(model_name, version, batch_label)
        atomic_pickle_dump(model, self.models_dir / artifact)

        entry: dict[str, Any] = {
            "version": f"v{version:04d}",
            "model": model_name,
            "batch_idx": batch_idx,
            "batch": batch_label,
            "created_at": _utc_now(),
            "artifact": artifact,
            "metrics": metrics,
            "context": context or {},
        }

        result = self.gate.evaluate(metrics, self.production_metrics(), context)
        entry["gate"] = result.to_dict()

        if self._should_promote(model_name, metrics, result):
            entry["status"] = "production"
            self._demote_current_production()
            self._state["production"] = entry
            logger.info(
                "Версия %s (%s, батч %d) назначена продуктовой",
                entry["version"], model_name, batch_idx,
            )
        else:
            entry["status"] = "rejected" if not result.passed else "candidate"
            if not result.passed:
                logger.info(
                    "Версия %s (%s) не прошла гейт: %s",
                    entry["version"], model_name, "; ".join(result.reasons),
                )

        self._state["versions"].append(entry)
        self._state.setdefault("latest", {})[model_name] = artifact
        self.save()
        return entry

    def _should_promote(
        self, model_name: str, metrics: dict[str, Any], result: GateResult
    ) -> bool:
        """Решить, становится ли версия продуктовой.

        Помимо прохождения гейта действует режим «выбирать лучшего»:
        если на батче есть несколько моделей, продуктовой становится
        лучшая по f1, а не просто первая прошедшая гейт. Иначе гейт
        мог бы закрепить за продуктивом посредственную модель.
        """
        if not result.passed:
            return False

        if not self.rules.get("promote_best_by_f1", True):
            return True

        production = self.production
        if production is None:
            return True

        if production.get("model") == model_name:
            # Та же модель: заменяем, только если стала заметно лучше.
            current = float((production.get("metrics") or {}).get("f1") or 0.0)
            candidate = float(metrics.get("f1") or 0.0)
            return candidate > current

        # Другая модель: переключаемся только при выигрыше по f1.
        current = float((production.get("metrics") or {}).get("f1") or 0.0)
        candidate = float(metrics.get("f1") or 0.0)
        return candidate > current

    def _demote_current_production(self) -> None:
        for entry in self._state["versions"]:
            if entry.get("status") == "production":
                entry["status"] = "previous"

    # -- указатели ------------------------------------------------------

    def refresh_pointers(self) -> dict[str, str]:
        """Обновить `best_model.pkl` и `<model>_latest.pkl`.

        Инференс читает именно эти файлы, поэтому он не должен знать
        о версионировании.
        """
        created: dict[str, str] = {}

        production = self.production
        if production:
            source = self.models_dir / production["artifact"]
            if source.is_file():
                import pickle

                with open(source, "rb") as handle:
                    model = pickle.load(handle)
                atomic_pickle_dump(model, self.models_dir / "best_model.pkl")
                created["best_model"] = production["artifact"]

        for model_name, artifact in self._state.get("latest", {}).items():
            source = self.models_dir / artifact
            if not source.is_file():
                continue
            import pickle

            with open(source, "rb") as handle:
                model = pickle.load(handle)
            atomic_pickle_dump(model, self.models_dir / f"{model_name}_latest.pkl")
            created[f"{model_name}_latest"] = artifact

        return created

    # -- очистка старых артефактов ---------------------------------------

    def prune(self, keep: int) -> dict[str, Any]:
        """Удалить файлы давно неактуальных версий, оставив записи реестра.

        Зачем это нужно. Версия на каждом батче — по четыре файла, и на
        48 батчах каталог `models/` вырастает примерно до 160 МБ, почти
        целиком из-за случайного леса (около 3 МБ на версию). В CI такой
        каталог уходит и в кэш состояния, и в артефакты при каждом
        запуске, хотя для продолжения обучения нужны лишь актуальные
        модели.

        Что и как хранится. Записи реестра — метрики, статус, причина
        отклонения — остаются **все**: история версий это и есть
        требование о хранении метрик по версиям. Удаляются только
        файлы, а в записи ставится `artifact_pruned: true`, чтобы
        отчёт не ссылался на несуществующий артефакт. Продуктовая
        версия и последние `keep` версий каждой модели не трогаются
        никогда.

        Args:
            keep: сколько последних версий каждой модели оставить.

        Returns:
            Сводка: удалённые файлы, освобождённые байты, сколько
            записей реестра осталось без файла.
        """
        if keep <= 0:
            # 0 или отрицательное значение выключает очистку: об этом
            # стоит сказать прямо, а не молча оставлять каталог растущим.
            logger.info("Очистка артефактов выключена: keep = %d", keep)
            return {"removed": [], "freed_bytes": 0, "pruned_entries": 0}

        by_model: dict[str, list[dict[str, Any]]] = {}
        for entry in self.versions:
            by_model.setdefault(str(entry.get("model", "")), []).append(entry)

        production_artifact = (self.production or {}).get("artifact")
        removed: list[str] = []
        freed = 0
        pruned_entries = 0

        for model_name, entries in by_model.items():
            # Версии нумеруются сквозным счётчиком, поэтому порядок в
            # реестре совпадает с порядком создания.
            outdated = entries[:-keep] if keep else []
            for entry in outdated:
                artifact = str(entry.get("artifact", ""))
                if not artifact or artifact == production_artifact:
                    # Продуктовую версию нельзя удалять даже если она
                    # старая: на неё ссылается инференс.
                    entry.setdefault("artifact_pruned", False)
                    continue
                if entry.get("artifact_pruned"):
                    # Файл уже удалён в прошлый вызов. Без этой
                    # проверки каждый батч сообщал бы об одних и тех же
                    # удалениях, и в логе это читалось бы как работа.
                    continue
                path = self.models_dir / artifact
                if path.is_file():
                    freed += path.stat().st_size
                    path.unlink()
                entry["artifact_pruned"] = True
                removed.append(artifact)
                pruned_entries += 1

        if removed:
            self.save()
            logger.info(
                "Удалено старых артефактов: %d (%.1f МБ); в реестре осталось "
                "%d записей без файла",
                len(removed), freed / (1024 * 1024), pruned_entries,
            )
        return {
            "removed": removed,
            "freed_bytes": freed,
            "pruned_entries": pruned_entries,
        }

    # -- сводка ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Краткая сводка по реестру для отчёта и манифеста."""
        versions = self.versions
        by_model: dict[str, int] = {}
        for entry in versions:
            by_model[entry["model"]] = by_model.get(entry["model"], 0) + 1

        rejected = [
            entry for entry in versions
            if entry.get("gate", {}).get("passed") is False
        ]

        return {
            "n_versions": len(versions),
            "by_model": by_model,
            # Сколько записей реестра остались без файла: важно, чтобы
            # по реестру было видно, что история не потеряна, а файлы
            # убраны осознанно.
            "n_artifacts_pruned": sum(
                1 for entry in versions if entry.get("artifact_pruned")
            ),
            "production": (
                {
                    "version": self.production["version"],
                    "model": self.production["model"],
                    "batch_idx": self.production["batch_idx"],
                    "f1": (self.production.get("metrics") or {}).get("f1"),
                }
                if self.production
                else None
            ),
            "rejected": len(rejected),
            "last_rejection": (
                {
                    "version": rejected[-1]["version"],
                    "model": rejected[-1]["model"],
                    "reasons": rejected[-1]["gate"].get("reasons", []),
                }
                if rejected
                else None
            ),
        }
