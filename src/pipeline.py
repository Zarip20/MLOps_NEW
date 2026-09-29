"""Оркестратор конвейера (АР-3).

`run.py` разбирает аргументы и вызывает методы этого класса; вся
последовательность этапов, условия и обработка ошибок сосредоточены здесь.
Причина — три требования, которые невозможно выполнить, размазав логику
по модулям:

* порядок этапов должен быть явным (в частности, ассоциативные правила
  должны появиться **до** проверки динамических правил — в исходном коде
  было наоборот, поэтому на первом батче метрика всегда была пустой, D-28);
* сбой любого этапа обязан приводить к возврату `False` и ненулевому коду
  выхода, а состояние не должно продвигаться (АР-11);
* прогон должен порождать манифест, по которому любой артефакт
  отслеживается до конкретного запуска (АР-6).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import association, data_collection
from src.collector import BatchCollector
from src.config import Config
from src.dashboard import build_dashboard
from src.data_quality import DataQualityEvaluator
from src.drift import DriftMonitor
from src.explain import feature_names_from_preprocessor, global_explanation
from src.features import engineer_features, profile_frame
from src.preprocessing import (
    create_preprocessor,
    feature_fingerprint,
    fingerprint_hash,
    input_feature_names,
)
from src.registry import ModelRegistry, QualityGate
from src.report import build_report
from src.rules import RuleSet
from src.state import StateStore
from src.storage import TrainingStore
from src.training import InsufficientDataError, load_models, save_model, train_batch
from src.utils import (
    Stopwatch,
    atomic_pickle_dump,
    describe_environment,
    load_json,
    parse_datetime,
    peak_memory_mb,
    safe_pickle_load,
    save_json,
    set_seed,
)

logger = logging.getLogger(__name__)

PREPROCESSOR_FILE = "preprocessor.pkl"
BEST_MODEL_FILE = "best_model.pkl"
COLLECTOR_FILE = "collector.pkl"


class PipelineError(RuntimeError):
    """Ошибка, из-за которой батч не может быть обработан."""


def _fmt_f1(value: Any) -> str:
    """Отформатировать f1 для журнала."""
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "н/д"


class Pipeline:
    """Оркестратор одного прогона конвейера."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.state = StateStore(config.state_file, root=config.root)
        self.watch = Stopwatch()
        set_seed(int(config.run.get("seed", 42)))

        quality_cfg = config.quality
        self.rules = RuleSet.from_config(quality_cfg["rules"])
        self.evaluator = DataQualityEvaluator(
            self.rules,
            max_row_missing_ratio=quality_cfg["max_row_missing_ratio"],
            max_violation_ratio=quality_cfg["max_violation_ratio"],
            feature_cols=config.feature_cols,
            numeric_cols=config.numerical_cols,
            expected_numeric=config.get("schema", {}).get("expected_numeric"),
        )

        store_cfg = config.get("training_store", {})
        # Хранилище держит ИСХОДНЫЕ признаки: производные восстанавливаются
        # при каждом обучении заново. Иначе при смене набора производных
        # признаков старое накопление оказалось бы несовместимым.
        # Временная колонка сохраняется в хранилище, потому что без неё
        # нельзя пересчитать календарные производные признаки.
        store_columns = [*config.feature_cols, config.target_name]
        store_columns += [
            column
            for column in [config.time_column, *config["batching"].get("context_columns", [])]
            if column not in store_columns
        ]
        self.store = (
            TrainingStore(
                config.path(store_cfg.get("path", "data_processed")),
                columns=store_columns,
            )
            if store_cfg.get("enabled", True)
            else None
        )

        self.engineering = config.get("feature_engineering", {})
        self.registry = ModelRegistry(
            config.models_dir, QualityGate(config.get("quality_gate", {}))
        )
        self.drift_cfg = config.get("drift", {})
        self.drift = DriftMonitor(self.drift_cfg)
        self.eda = config.get("eda", {})
        self._numerical_features = list(config.numerical_cols)
        self._engineered: list[str] = []

    @property
    def numerical_features(self) -> list[str]:
        """Числовые признаки, включая производные."""
        return list(self._numerical_features)

    def _refresh_features(self, added: list[str] | None = None) -> None:
        """Зафиксировать состав числовых признаков.

        Первый батч определяет, какие производные признаки появились.
        Дальнейшие батчи обязаны использовать тот же набор: если бы
        ширина матрицы менялась от батча к батчу, `partial_fit` сломался бы
        на первой же несовпадности.
        """
        if added:
            for name in added:
                if name not in self._engineered:
                    self._engineered.append(name)
        self._numerical_features = list(self.config.numerical_cols) + list(self._engineered)

    # ------------------------------------------------------------------
    # Инициализация
    # ------------------------------------------------------------------

    def init(self, reset: bool = False) -> dict[str, Any]:
        """Распаковать источник, разбить на батчи, зафиксировать состояние.

        Заодно сохраняется сборщик данных (`collector.pkl`): это отдельный
        сериализуемый артефакт, который выгружается в CI и позволяет принять
        новый батч в другом прогоне (задание 2, пункт 2.b.ii).

        Args:
            reset: начать с нуля, даже если данные не изменились. Нужен
                для осознанного полного переобучения.

        Returns:
            Сводка: сколько батчей, границы, хэши и сведения о том,
            сохранился ли прогресс.

        **Почему прогресс по умолчанию сохраняется.** `init` вызывается
        на каждом запуске CI, в том числе по расписанию (пункт 3.5
        задания 2). Набор батчей при этом не меняется, а каталог с ними
        каждый раз создаётся заново: `data/raw_batches/` не входит в
        кэш состояния. Если бы `init` сбрасывал прогресс, каждое
        срабатывание расписания обучало бы систему заново с первого
        батча и дообучение было бы невозможно.

        Прогресс сбрасывается в двух случаях, и оба означают, что старые
        номера батчей больше ничего не значат: набор данных изменился
        (не совпал хэш) либо сброс запрошен явно.
        """
        dataset_hash = data_collection.dataset_hash(self.config)
        same_data = (
            self.state.is_initialised
            and dataset_hash is not None
            and self.state.dataset_hash == dataset_hash
        )
        reused = False

        if same_data and not reset:
            missing = self.state.missing_batches()
            if not missing:
                # Всё на месте: повторное разбиение не нужно, а прогресс
                # трогать нельзя.
                logger.info(
                    "Набор данных не изменился (хэш %s), батчей: %d; "
                    "уже обработано: %d",
                    str(dataset_hash)[:12], len(self.state),
                    self.state.last_processed + 1,
                )
                reused = True
            else:
                logger.info(
                    "Отсутствуют файлы %d батчей — выполняется повторное "
                    "разбиение без сброса прогресса", len(missing),
                )

        if reused:
            BatchCollector.from_config(self.config).save(
                self.config.models_dir / COLLECTOR_FILE
            )
            return self._init_summary(reused=True)

        if self.state.is_initialised and not same_data and not reset:
            logger.warning(
                "Набор данных изменился (был %s, стал %s) — прогресс "
                "обработки сброшен",
                str(self.state.dataset_hash)[:12], str(dataset_hash)[:12],
            )

        with self.watch.measure("init"):
            source = data_collection.extract_source(self.config)
            batches = data_collection.split_into_batches(
                self.config, source, self.state,
                # Сброс нужен только при явном требовании: смена набора
                # данных и так делает список батчей другим, а повторное
                # разбиение тех же данных оставляет нумерацию верной.
                reset_progress=reset or not same_data,
            )

        collector = BatchCollector.from_config(self.config)
        collector.save(self.config.models_dir / COLLECTOR_FILE)

        summary = self._init_summary(reused=False)
        summary["batches"] = len(batches)
        summary["collector"] = COLLECTOR_FILE
        return summary

    def _init_summary(self, reused: bool) -> dict[str, Any]:
        """Сводка по результату инициализации."""
        batches = self.state.batches
        return {
            "batches": len(batches),
            "first": batches[0] if batches else None,
            "last": batches[-1] if batches else None,
            "processed": self.state.last_processed + 1,
            "reused": reused,
            "dataset_hash": data_collection.dataset_hash(self.config),
            "config_hash": self.config.config_hash,
            "collector": COLLECTOR_FILE,
        }

    # ------------------------------------------------------------------
    # Обработка батчей
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Синхронизация с внешними данными
    # ------------------------------------------------------------------

    def sync_batches(self) -> list[str]:
        """Добавить в состояние батчи, принятые сборщиком извне.

        `init` фиксирует список батчей один раз. Если позже в каталог
        положить новый файл — например, через `python -m src.collector
        --append …`, — конвейер его не увидит: состояние о нём не знает,
        и `update` такой батч просто пропустит. Здесь список из
        артефакта сверяется с состоянием, а новые батчи дописываются в
        конец, чтобы уже обработанные батчи не сдвинулись.

        Returns:
            Имена добавленных батчей; пустой список, если новых нет.
        """
        collector = BatchCollector.from_config(self.config)
        known = set(self.state.batches)
        prefix = self.config.paths.get("data_raw", "data/raw_batches")

        discovered: list[tuple[str, str]] = []
        for name in collector.batch_names():
            relative = str(Path(prefix) / name)
            if relative not in known:
                discovered.append((name, relative))

        if not discovered:
            return []

        discovered.sort()
        self.state.add_batches(
            [relative for _, relative in discovered],
            dataset_hash=data_collection.dataset_hash(self.config),
        )
        names = [name for name, _ in discovered]
        logger.info(
            "Синхронизировано с каталогом: добавлено батчей %d (%s … %s), "
            "всего батчей %d",
            len(names), names[0], names[-1], len(self.state),
        )
        return names

    def update(self, limit: int | None = None) -> int:
        """Обработать очередные батчи.

        Args:
            limit: максимум батчей за один вызов. `None` — все оставшиеся.

        Returns:
            Число успешно обработанных батчей.
        """
        if not self.state.is_initialised:
            raise PipelineError(
                "Батчи не инициализированы. Сначала выполните `-mode init`."
            )

        # Батчи, принятые сборщиком извне, добавляются в состояние до
        # подсчёта остатка, иначе они не попадут в этот прогон.
        self.sync_batches()

        remaining = self.state.remaining()
        if limit is not None:
            remaining = min(remaining, max(int(limit), 0))
        if remaining == 0:
            logger.info("Новых батчей для обработки нет")
            return 0

        processed = 0
        started = time.perf_counter()

        for index in range(self.state.last_processed + 1, self.state.last_processed + 1 + remaining):
            label = data_collection.batch_label(self.state, index)
            logger.info("── Батч %d (%s) ──", index, label)
            try:
                self._process_batch(index)
            except Exception as error:  # noqa: BLE001 — причина логируется ниже
                # Состояние не продвигается: прерванный батч будет обработан
                # повторно при следующем запуске (АР-11).
                logger.exception("Батч %d (%s) не обработан: %s", index, label, error)
                logger.info(
                    "Обработано батчей до сбоя: %d из %d. Повторный запуск "
                    "продолжит с батча %d.",
                    processed, remaining, index,
                )
                return processed

            processed += 1
            self.state.mark_processed(
                index,
                dataset_hash=data_collection.dataset_hash(self.config),
                config_hash=self.config.config_hash,
                label=label,
            )

        elapsed = time.perf_counter() - started
        logger.info(
            "Обработано батчей: %d за %.1f с (%.2f с/батч)",
            processed, elapsed, elapsed / processed if processed else 0.0,
        )
        return processed

    def _process_batch(self, index: int) -> dict[str, Any]:
        """Полный цикл обработки одного батча."""
        config = self.config
        label = data_collection.batch_label(self.state, index)
        # Снимок времени нужен, чтобы в манифест попала длительность
        # этого батча, а не накопленная с начала прогона.
        timing = self.watch.snapshot()

        # 1. Чтение сырого батча.
        with self.watch.measure("read"):
            path = data_collection.batch_path(config, self.state, index)
            raw = data_collection.read_batch(path, config)

        # 2. Качество данных по сырым данным: полнота, валидность, типы,
        #    своевременность, статические правила.
        with self.watch.measure("quality"):
            quality = self.evaluator.evaluate(
                raw,
                config.time_column,
                dynamic_rules=[],
                date_format=config["batching"].get("date_format"),
            )

        # 3. Очистка.
        with self.watch.measure("clean"):
            cleaned, cleaning = self.evaluator.clean_data(raw)
        quality["cleaning"] = cleaning
        logger.info(
            "Очистка: %d → %d строк (удалено %d, %.2f %%)",
            cleaning["rows_in"], cleaning["rows_out"],
            cleaning["rows_removed"], cleaning["removed_ratio"] * 100,
        )

        if cleaned.empty:
            raise PipelineError(
                f"После очистки батча {label} не осталось ни одной строки — "
                f"проверьте правила качества в config.yaml"
            )

        # 4. Целевая метка и разведочный анализ.
        target_frame = self._build_target(cleaned)
        quality["target"] = self._target_summary(target_frame)

        with self.watch.measure("eda"):
            quality["eda"] = self._run_eda(raw, target_frame)

        # 5. Ассоциативные правила. Генерируются один раз, на первом батче,
        #    но ДО проверки динамических правил — иначе на первом батче
        #    метрика всегда была бы пустой (D-28).
        with self.watch.measure("association"):
            rules = association.load_association_rules(config.rules_file)
            if not rules:
                parameters = config.association
                rules = association.generate_association_rules(
                    target_frame,
                    config.categorical_cols,
                    config.rules_file,
                    min_support=parameters["min_support"],
                    min_confidence=parameters["min_confidence"],
                    min_lift=parameters["min_lift"],
                    max_len=parameters.get("max_len"),
                    n_rules=parameters["n_rules"],
                    algorithm=parameters.get("algorithm", "fpgrowth"),
                    exclude_majority_consequent=parameters.get(
                        "exclude_majority_consequent", True
                    ),
                )
                logger.info("Найдено ассоциативных правил: %d", len(rules))
        quality["association"] = association.rule_statistics(rules)

        # 6. Динамические правила — теперь по актуальным правилам.
        with self.watch.measure("quality_dynamic"):
            quality["dynamic_rule_violations"] = self.evaluator.check_dynamic_rules(
                raw,
                rules,
                min_antecedent_rows=int(
                    config.association.get("min_antecedent_rows", 30)
                ),
            )

        # 6б. Дрейф данных относительно обучающего окна (2.b.iv).
        #     Конвейер при этом не прерывается: результат влияет на
        #     решение гейта качества, а не на успешность прогона (АР-10).
        with self.watch.measure("drift"):
            drift = self._check_drift(raw, target_frame)
        quality["drift"] = drift

        # 7. Производные признаки. Добавляются ДО предобработки, поэтому
        #    проходят через тот же ColumnTransformer (2.b.ii).
        with self.watch.measure("features"):
            target_frame, added = self._engineer(target_frame)
        self._refresh_features(added)
        quality["features"] = {
            "engineered": added,
            "numerical_total": len(self.numerical_features),
        }

        # 8. Обучение и валидация. Разделение train/val, обучение
        #    препроцессора на train-части и обучение моделей выполняются
        #    внутри train_batch — порядок шагов там жёсткий, нарушать его
        #    нельзя, иначе валидация снова увидит обучающие данные.
        with self.watch.measure("train"):
            existing = load_models(config)
            outcome = train_batch(
                config,
                target_frame,
                self.store,
                existing,
                preprocessor=self._load_preprocessor(),
                numerical_cols=self.numerical_features,
                engineer=self._store_engineer,
            )

        # 9. Реестр версий и гейт качества (АР-5). Все версии сохраняются,
        #    независимо от результата гейта: регресс должен остаться в
        #    истории, иначе непонятно, почему качество упало.
        with self.watch.measure("register"):
            registration = self._register_models(index, label, outcome, drift)

        # 10. Препроцессор и указатели на модели.
        with self.watch.measure("save"):
            atomic_pickle_dump(
                outcome.preprocessor, config.models_dir / PREPROCESSOR_FILE
            )
            self.registry.refresh_pointers()
            # Старые версии удаляются после обновления указателей:
            # `best_model.pkl` и `<model>_latest.pkl` уже перезаписаны
            # с актуальных файлов, и очистка их не затронет, даже если
            # продуктовая версия старая.
            self._prune_models()

        # 11. Накопление обучающей части (только train — валидация в обучение
        #     не попадает никогда).
        with self.watch.measure("store"):
            stored = 0
            if self.store is not None and outcome.train_part is not None:
                stored = self.store.append(index, outcome.train_part)

        # 12. Интерпретация моделей (5.b.i).
        with self.watch.measure("explain"):
            explanation = self._explain(outcome)

        # 13. Метаданные и манифест.
        self._write_metadata(
            index, label, quality, outcome, outcome.preprocessor, stored, rules, timing,
            registration=registration, explanation=explanation,
        )

        best_name = registration.get("best_model")
        best_entry = registration.get("best_version")
        production = self.registry.production or {}
        best_f1 = (outcome.metrics.get(best_name) or {}).get("f1") if best_name else None
        logger.info(
            "Батч %d готов: лучшая по f1 — %s %s (f1 = %s); продуктовая — %s %s",
            index,
            best_name or "—",
            best_entry or "",
            _fmt_f1(best_f1),
            production.get("model", "—"),
            production.get("version", ""),
        )
        for note in outcome.notes:
            logger.info("  · %s", note)
        for entry in registration.get("rejected", []):
            logger.info("  · %s", entry)
        if drift.get("status") in ("warn", "alert"):
            logger.info(
                "  · дрейф: %s (max PSI = %s)", drift.get("status"), drift.get("max_psi")
            )

        return {
            "best_model": best_name,
            "metrics": outcome.metrics,
            "production": production,
        }

    def _run_eda(self, raw: pd.DataFrame, target_frame: pd.DataFrame) -> dict[str, Any]:
        """Разведочный анализ батча (2.b.i)."""
        if not self.eda.get("enabled", True):
            return {"enabled": False}

        profile = profile_frame(
            raw,
            numerical_cols=self.config.numerical_cols,
            categorical_cols=self.config.categorical_cols,
            target=None,
        )
        profile["target"] = self._target_summary(target_frame)
        profile["enabled"] = True
        return profile

    def _check_drift(
        self, raw: pd.DataFrame, target_frame: pd.DataFrame
    ) -> dict[str, Any]:
        """Проверить дрейф относительно обучающего окна (2.b.iv).

        Дрейф оценивается на очищенном батче, а не на сыром: модель
        никогда не видит неочищенные данные, и сдвиг в очищенном потоке
        — это ровно тот сигнал, который влияет на её работу. Проблемы,
        связанные с очисткой, фиксируются отдельно в отчёте очистки.
        """
        if not self.drift_cfg.get("enabled", True):
            return {
                "available": False,
                "status": "unknown",
                "reason": "мониторинг дрейфа выключен",
            }

        target = self.config.target_name
        if not self.drift.has_reference():
            self.drift.set_reference(
                self.config.numerical_cols,
                self.config.categorical_cols,
                target,
                target_frame,
            )
            return {
                "available": True,
                "status": "baseline",
                "note": "эталон дрейфа построен по текущему батчу",
            }

        result = self.drift.check(
            target_frame,
            target,
            self.config.numerical_cols,
            self.config.categorical_cols,
        )
        self.drift.extend_reference(
            target_frame,
            self.config.numerical_cols,
            self.config.categorical_cols,
            target,
            window_batches=int(self.drift_cfg.get("reference_window", 3)),
        )
        return result

    def _engineer(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Добавить производные признаки (2.b.ii)."""
        if not self.engineering.get("enabled", True):
            return frame, []

        # Первый батч задаёт состав производных признаков; последующие
        # обязаны дать тот же набор, иначе изменится ширина матрицы
        # и сломается `partial_fit`.
        expected = self._engineered
        if expected:
            # Состав производных признаков зафиксирован на первом батче.
            result, _ = engineer_features(
                frame,
                self.config.numerical_cols,
                self.config.categorical_cols,
                time_column=self.config.time_column,
            date_format=self.config["batching"].get("date_format"),
            )
            return result, []

        result, added = engineer_features(
            frame,
            self.config.numerical_cols,
            self.config.categorical_cols,
            time_column=self.config.time_column,
            date_format=self.config["batching"].get("date_format"),
        )
        self._engineered = added
        return result, added

    def _sync_features_from_preprocessor(self, preprocessor: Any) -> None:
        """Синхронизировать набор признаков со схемой препроцессора.

        Обязательный шаг при загрузке сохранённого препроцессора. Без
        него возобновление прогона в новом процессе и `inference`
        подавали бы препроцессору матрицу другой ширины: набор
        производных признаков фиксируется на первом батче и из
        конфигурации сам не восстанавливается.
        """
        expected = input_feature_names(preprocessor)
        if not expected.get("num"):
            return

        derived = [
            name for name in expected["num"]
            if name not in self.config.numerical_cols
        ]
        changed = list(self.numerical_features) != list(expected["num"])
        self._numerical_features = list(expected["num"])
        if derived != self._engineered:
            self._engineered = derived
        if changed:
            logger.info(
                "Схема признаков взята из препроцессора: числовых %d, "
                "из них производных %d",
                len(expected["num"]), len(derived),
            )

    def _store_engineer(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Привести данные накопительного хранилища к схеме признаков.

        Хранилище содержит исходные колонки намеренно: состав производных
        признаков может быть переопределён в конфигурации, и тогда старое
        накопление пришлось бы пересчитывать. Здесь применяется тот же
        штатный инжиниринг, что и к текущему батчу, — иначе матрицы
        получили бы разный набор колонок.
        """
        if not self._engineered:
            return frame
        result, _ = engineer_features(
            frame,
            self.config.numerical_cols,
            self.config.categorical_cols,
            time_column=self.config.time_column,
            date_format=self.config["batching"].get("date_format"),
        )
        return result

    def _prune_models(self) -> dict[str, Any]:
        """Удалить файлы давно неактуальных версий моделей.

        Вызывается после каждого батча: без ограничения каталог
        `models/` растёт на четыре файла за батч и на полном наборе
        занимает около 160 МБ, которые уезжали бы в CI и в кэш
        состояния. Записи реестра при этом сохраняются все — см.
        `ModelRegistry.prune`.
        """
        keep = int(
            self.config.get("registry", {}).get("keep_artifacts_per_model", 5)
        )
        return self.registry.prune(keep)

    def _register_models(
        self,
        index: int,
        label: str,
        outcome: Any,
        drift: dict[str, Any],
    ) -> dict[str, Any]:
        """Зарегистрировать версии моделей и применить гейт качества."""
        psi = drift.get("max_psi")
        context = {
            "psi": psi,
            "drift_status": drift.get("status"),
        }

        entries: list[dict[str, Any]] = []
        for name, model in outcome.models.items():
            entry = self.registry.register(
                model=model,
                model_name=name,
                batch_idx=index,
                batch_label=label,
                metrics=outcome.metrics[name],
                context=context,
            )
            entries.append(entry)

        best = max(
            entries,
            key=lambda item: (item.get("metrics") or {}).get("f1") or 0.0,
            default=None,
        )

        rejected = [
            f"{entry['model']} {entry['version']} не прошёл гейт: "
            + "; ".join(entry["gate"]["reasons"])
            for entry in entries
            if not entry["gate"]["passed"]
        ]

        production = self.registry.production or {}
        return {
            "best_model": best["model"] if best else None,
            "best_version": best["version"] if best else None,
            "production": {
                "model": production.get("model"),
                "version": production.get("version"),
                "f1": (production.get("metrics") or {}).get("f1"),
            },
            "registered": [entry["version"] for entry in entries],
            "rejected": rejected,
            "summary": self.registry.summary(),
        }

    def _explain(self, outcome: Any) -> dict[str, Any]:
        """Собрать интерпретацию всех обученных моделей (5.b.i)."""
        if not self.config.get("explain", {}).get("enabled", True):
            return {"enabled": False}

        names = feature_names_from_preprocessor(outcome.preprocessor)
        if not names:
            names = self.numerical_features + self.config.categorical_cols
        return global_explanation(outcome.models, names)

    def _build_target(self, cleaned: pd.DataFrame) -> pd.DataFrame:
        """Сформировать целевую метку и оставить признаки.

        Временная колонка сохраняется в результирущем фрейме, хотя в
        признаки не входит: она нужна для вычисления производных
        признаков (возраст автомобиля, календарные признаки). В
        `train_batch` отбираются только колонки-признаки, поэтому
        лишний столбец в обучение не попадает.
        """
        config = self.config
        source = config.target_column
        if source not in cleaned.columns:
            raise PipelineError(f"В батче нет колонки с меткой {source!r}")

        frame = cleaned.copy()
        if config.data["target"].get("positive_rule", "notna") == "notna":
            frame[config.target_name] = frame[source].notna().astype(int)
        else:
            frame[config.target_name] = (
                pd.to_numeric(frame[source], errors="coerce").fillna(0) > 0
            ).astype(int)

        missing = [column for column in config.feature_cols if column not in frame.columns]
        if missing:
            raise PipelineError(f"В батче отсутствуют признаки: {missing}")

        keep = [*config.feature_cols, config.target_name]
        context = [config.time_column, *config["batching"].get("context_columns", [])]
        for column in context:
            if column in frame.columns and column not in keep:
                keep.append(column)
        return frame[keep]

    def _target_summary(self, cleaned: pd.DataFrame) -> dict[str, Any]:
        """Распределение метки — база для отслеживания дрейфа."""
        positive = int(cleaned[self.config.target_name].sum())
        total = max(len(cleaned), 1)
        return {
            "rows": int(len(cleaned)),
            "positive": positive,
            "positive_rate": round(positive / total, 6),
        }

    def _load_preprocessor(self) -> Any | None:
        """Загрузить сохранённый препроцессор, если он есть и читается.

        Отсутствие или повреждение файла не является ошибкой: препроцессор
        будет обучен заново на train-части текущего батча.
        """
        path = self.config.models_dir / PREPROCESSOR_FILE
        if not path.is_file():
            logger.info("Препроцессор не найден — будет обучен на текущем батче")
            return None
        try:
            preprocessor = safe_pickle_load(path)
        except ValueError as error:
            logger.warning("%s", error)
            logger.info("Препроцессор будет переобучен на текущем батче")
            return None
        logger.info("Препроцессор загружен из %s", path.name)
        self._sync_features_from_preprocessor(preprocessor)
        return preprocessor

    def _write_metadata(
        self,
        index: int,
        label: str,
        quality: dict[str, Any],
        outcome: Any,
        preprocessor: Any,
        stored: int,
        rules: list[dict[str, Any]],
        timing: dict[str, float],
        registration: dict[str, Any] | None = None,
        explanation: dict[str, Any] | None = None,
    ) -> None:
        """Сохранить метаданные качества, метрик и манифест прогона."""
        config = self.config
        metadata_dir = config.metadata_dir

        save_json(metadata_dir / f"quality_{index:04d}.json", {"batch_idx": index, **quality})
        save_json(
            metadata_dir / f"model_metrics_{index:04d}.json",
            {"batch_idx": index, "batch": label, **outcome.metrics},
        )

        fingerprint = feature_fingerprint(preprocessor)
        registry_view = self.registry.summary()
        registry_view["versions"] = [
            {
                "version": entry["version"],
                "model": entry["model"],
                "batch_idx": entry["batch_idx"],
                "f1": (entry.get("metrics") or {}).get("f1"),
                "status": entry.get("status"),
                "reasons": (entry.get("gate") or {}).get("reasons", []),
            }
            for entry in self.registry.versions
        ]

        manifest = {
            "batch_idx": index,
            "batch": label,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": config.fingerprint(),
            "environment": describe_environment(),
            "data": {
                "rows_raw": quality.get("rows"),
                "rows_clean": quality.get("cleaning", {}).get("rows_out"),
                "rows_stored": stored,
                "target": quality.get("target"),
            },
            "training": {
                "n_train": outcome.n_train,
                "n_val": outcome.n_val,
                "positive_rate_train": outcome.positive_rate_train,
                "positive_rate_val": outcome.positive_rate_val,
                "durations": outcome.durations,
                "best_model": (registration or {}).get("best_model"),
                "notes": outcome.notes,
            },
            "metrics": outcome.metrics,
            "registry": registry_view,
            "preprocessor": {**fingerprint, "fingerprint_hash": fingerprint_hash(fingerprint)},
            "features": quality.get("features", {}),
            "drift": quality.get("drift", {}),
            "association": quality.get("association", {}),
            "explanation": explanation or {},
            "performance": {
                "durations": self.watch.delta(timing),
                "batch_total_seconds": round(
                    sum(self.watch.delta(timing).values()), 3
                ),
                **peak_memory_mb(),
            },
            "training_store": self.store.stats() if self.store else {"enabled": False},
        }
        save_json(metadata_dir / f"run_manifest_{index:04d}.json", manifest)

    # ------------------------------------------------------------------
    # Инференс и отчёт
    # ------------------------------------------------------------------

    def inference(self, file_path: str | Path) -> Path:
        """Применить лучшую модель к новым данным."""
        config = self.config
        source = Path(file_path)
        if not source.is_file():
            raise FileNotFoundError(f"Файл с данными не найден: {source}")

        preprocessor_path = config.models_dir / PREPROCESSOR_FILE
        model_path = config.models_dir / BEST_MODEL_FILE
        if not preprocessor_path.is_file() or not model_path.is_file():
            raise PipelineError(
                "Модель не найдена. Сначала обработайте хотя бы один батч: `-mode update`"
            )

        # Имена колонок исходного файла запоминаются ДО подготовки
        # признаков, чтобы в результат не попали служебные.
        original_columns = list(pd.read_csv(source, nrows=0).columns)
        frame = pd.read_csv(source, low_memory=False)
        missing = [column for column in config.feature_cols if column not in frame.columns]
        if missing:
            raise PipelineError(f"В файле для инференса нет признаков: {missing}")

        preprocessor = safe_pickle_load(preprocessor_path)
        model = safe_pickle_load(model_path)

        # Набор производных признаков восстанавливается из самого
        # препроцессора: он обучается на первом батче, и пересборка
        # списка из конфигурации дала бы матрицу другой ширины.
        self._sync_features_from_preprocessor(preprocessor)

        time_column = config.time_column
        if time_column in frame.columns:
            frame[time_column] = parse_datetime(
                frame[time_column], config["batching"].get("date_format")
            )
        if self._engineered:
            frame, _ = engineer_features(
                frame,
                config.numerical_cols,
                config.categorical_cols,
                time_column=time_column,
                date_format=config["batching"].get("date_format"),
            )

        required = [*self.numerical_features, *config.categorical_cols]
        absent = [column for column in required if column not in frame.columns]
        if absent:
            raise PipelineError(
                f"После подготовки признаков не хватает колонок: {absent}"
            )

        matrix = preprocessor.transform(frame[required])
        # В результат попадают только исходные колонки плюс прогноз:
        # производные признаки — служебные и пользователю не нужны.
        output = frame[original_columns].copy() if original_columns else frame.copy()
        output["predict"] = model.predict(matrix)
        if hasattr(model, "predict_proba"):
            output["predict_proba"] = model.predict_proba(matrix)[:, 1]

        output_dir = source.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"{source.stem}_with_predict{source.suffix}"
        output.to_csv(target, index=False)
        logger.info("Результат сохранён: %s", target.name)
        return target

    def summary(self) -> tuple[Path, Path]:
        """Собрать текстовый отчёт и дашборд по истории прогонов.

        Returns:
            (путь к текстовому отчёту, путь к дашборду).
        """
        report = build_report(self.config, self.state)
        dashboard = build_dashboard(self.config.reports_dir, self.config.metadata_dir)
        return report, dashboard
