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
from src.config import Config, ConfigError
from src.dashboard import build_dashboard
from src.inference import SafePredictor, remember_training_ranges, validate_input
from src.meta import analyse as analyse_meta
from src.data_quality import DataQualityEvaluator
from src.drift import DriftMonitor
from src.explain import feature_names_from_preprocessor, global_explanation
from src.features import engineer_features, profile_frame
from src.preprocessing import (
    create_preprocessor,
    feature_fingerprint,
    fingerprint_hash,
    input_feature_names,
    known_categories,
)
from src.preprocessing_sweep import (
    evaluate as evaluate_variants,
)
from src.preprocessing_sweep import (
    pick as pick_variant,
)
from src.preprocessing_sweep import (
    should_run as should_sweep,
)
from src.registry import ModelRegistry, QualityGate
from src.report import build_report
from src.rules import RuleSet
from src.state import StateStore
from src.views import PipelineContext, default_registry
from src.storage import TrainingStore
from src.training import (
    InsufficientDataError,
    constructor_args,
    load_models,
    save_model,
    split_batch,
    train_batch,
)
from src.utils import (
    Stopwatch,
    atomic_pickle_dump,
    describe_environment,
    load_json,
    parse_datetime,
    peak_memory_mb,
    read_artifacts,
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
    """Оркестратор одного прогона конвейера.

    Находится в слое контроллера: знает порядок этапов и ничего не
    знает о том, как считается качество, обучаются модели и строится
    вывод. Доменные вычисления живут в отдельных модулях, вывод
    вынесен за порты представлений (7.b.iv).
    """

    def __init__(self, config: Config, views: Any = None) -> None:
        self.config = config
        self.state = StateStore(config.state_file, root=config.root)
        self.watch = Stopwatch()
        set_seed(int(config.run.get("seed", 42)))

        # Источники данных проверяются здесь, а не в `init`: неверный
        # список источников должен быть виден на старте, а не через
        # минуту, когда конвейер уже распаковал архив и нарезал батчи.
        try:
            self.sources = data_collection.sources_from_config(config)
        except data_collection.SourceError as error:
            raise ConfigError(
                f"Некорректный список источников данных: {error}"
            ) from error

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

        # Слой вывода подменяется снаружи: в тестах достаточно подставить
        # объект с методом `render`, а в бою работает набор по умолчанию.
        self.views = views or default_registry()

        # Обучающие диапазоны для проверки входа при инференсе. Пусто
        # до первого обучения: проверять нечего, и пропуск честнее,
        # чем молчаливые границы, взятые из первого же файла.
        self._training_ranges: dict[str, tuple[float, float]] = {}

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
            batches = data_collection.split_all_sources(
                self.config, self.state,
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

        # Обучающие диапазоны для проверки входа (6.b.ii). Берутся из
        # батча, на котором модель только что обучилась: брать их из
        # всех накопленных данных значило бы сдвигать «нормальный»
        # диапазон вслед за дрейфом, и проверка перестала бы замечать
        # именно то, ради чего заведена.
        self._training_ranges = remember_training_ranges(
            target_frame, self.numerical_features
        )

        # 8.1. Перебор вариантов предобработки (3.b.ii). Запускается на
        # батче, где препроцессор обучается заново; выбор записывается и
        # в отчёт, и в манифест, чтобы было видно, на чём он сделан.
        if should_sweep(config, index):
            quality["preprocessing_sweep"] = self._sweep_preprocessing(
                index, target_frame
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

    def _sweep_preprocessing(
        self, index: int, frame: pd.DataFrame
    ) -> dict[str, Any]:
        """Перебрать варианты предобработки и выбрать лучший (3.b.i).

        Перебор обучается заново на train-части текущего батча и
        оценивается на val-части — той же, что и обычное обучение, иначе
        сравнение вариантов шло бы по разным данным.
        """
        config = self.config
        sweep_cfg = config.get("preprocessing", {}).get("sweep", {})
        model_name = str(sweep_cfg.get("model", "lr"))
        metric = str(sweep_cfg.get("metric", "f1"))

        train_part, val_part, notes = split_batch(
            frame, config.validation,
            config.validation.get("min_positive_samples", 50),
            target=config.target_name,
        )
        results = evaluate_variants(
            config,
            frame,
            train_part,
            val_part,
            config.target_name,
            self.numerical_features,
            config.categorical_cols,
            model_name=model_name,
            model_params=constructor_args(config, model_name),
        )
        winner = pick_variant(results, metric)
        logger.info(
            "Перебор предобработки на батче %d: %d вариантов, победил %s "
            "(%s = %s)",
            index, len(results),
            winner.get("name", "—"), metric, winner.get(metric, "—"),
        )
        if winner.get("status") == "none":
            logger.warning(
                "Ни один вариант предобработки не построился, остаётся "
                "базовая конфигурация"
            )

        payload = {
            "batch_idx": index,
            "model": model_name,
            "metric": metric,
            "results": results,
            "best": winner,
            "notes": notes,
        }
        # Отдельным отчётом: перебор нужен не для обучения, а чтобы
        # решение о предобработке можно было проверить и оспорить.
        save_json(self.config.reports_dir / "preprocessing_sweep.json", payload)
        return payload

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
            "hyperparameters": outcome.hyperparameters,
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
        """Применить модель к новым данным с проверкой и калибровкой.

        Отличие от прежней версии: вход проверяется **до** обращения к
        модели, решения принимаются по калиброванному порогу, а при
        неприменимости продуктовой модели прогноз уходит запасной, а не
        прерывается (6.b.ii).

        Returns:
            Путь к файлу с исходными колонками и прогнозом.
        """
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

        preprocessor = safe_pickle_load(preprocessor_path)
        best_model = safe_pickle_load(model_path)

        # Набор производных признаков восстанавливается из самого
        # препроцессора: он обучается на первом батче, и пересборка
        # списка из конфигурации дала бы матрицу другой ширины.
        self._sync_features_from_preprocessor(preprocessor)

        # Проверка входа — до любых преобразований: чтобы понять, что
        # файлу не хватает колонок, не нужно сначала ломаться на
        # препроцессоре.
        validation = validate_input(
            frame,
            required_columns=config.feature_cols,
            numerical_columns=config.numerical_cols,
            known_categories=(
                self._known_categories(preprocessor) if (
                    config.inference_config.get("validate", {}).get("check_categories", True)
                ) else None
            ),
            # Диапазоны передаются явно, а не берутся из накопленных при
            # обучении: проверка входа не должна зависеть от того, что
            # успел накопить процесс, — иначе результат менялся бы от
            # порядка запусков.
            ranges=self._training_ranges,
            max_outlier_ratio=float(
                config.inference_config.get("validate", {}).get(
                    "max_outlier_ratio", 0.2
                )
            ),
        )
        if not validation.accepted:
            raise PipelineError(
                "Файл для инференса непригоден: " + "; ".join(validation.problems)
            )
        if validation.degraded:
            logger.warning(
                "Входные данные требуют оговорок: %s", "; ".join(validation.problems)
            )

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

        # Кандидаты: продуктовая и запасная. Набор берётся из указателей
        # `*_latest.pkl`, которые обновляются на каждом батче, поэтому
        # запасная модель соответствует текущей схеме признаков.
        models: dict[str, Any] = {"best": best_model}
        fallback_name = config.inference_fallback_model
        fallback_path = config.models_dir / f"{fallback_name}_latest.pkl"
        if fallback_path.is_file():
            try:
                models[fallback_name] = safe_pickle_load(fallback_path)
            except ValueError as error:
                logger.warning(
                    "Запасная модель %s не прочитана: %s", fallback_name, error
                )

        target_rate = (
            config.target_positive_rate()
            if config.inference_threshold_mode == "quantile" else None
        )
        predictor = SafePredictor(
            models,
            preprocessor,
            required,
            fallback=fallback_name,
            threshold_mode=config.inference_threshold_mode,
            target_positive_rate=target_rate,
        )
        result = predictor.predict(frame, validation)

        # В результат попадают только исходные колонки плюс прогноз:
        # производные признаки — служебные и пользователю не нужны.
        output = frame[original_columns].copy() if original_columns else frame.copy()
        output["predict"] = np.asarray(result.predictions, dtype=int)
        if result.probabilities is not None:
            output["predict_proba"] = np.round(
                np.asarray(result.probabilities, dtype=float), 6
            )

        output_dir = source.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        target_path = output_dir / f"{source.stem}_with_predict{source.suffix}"
        output.to_csv(target_path, index=False)

        report = result.as_dict()
        report["source"] = str(source)
        report["target_positive_rate"] = target_rate
        save_json(output_dir / f"{source.stem}_inference.json", report)

        for note in result.notes:
            logger.info("Инференс: %s", note)
        share = (
            100.0 * float(np.mean(result.predictions))
            if len(result.predictions) else 0.0
        )
        logger.info(
            "Прогноз: модель %s, положительных %.2f %%, порог %s",
            result.model, share,
            "не применяется" if result.threshold is None
            else f"{result.threshold:.4f}",
        )
        logger.info("Результат сохранён: %s", target_path.name)
        return target_path

    def _known_categories(self, preprocessor: Any) -> dict[str, list[str]]:
        """Значения категорий, известные по обученному препроцессору.

        Источник — обученный `OneHotEncoder`, а не метаданные качества:
        там хранится лишь несколько самых частых значений, и для `MAKE`
        с сотнями марок этого хватило бы на то, чтобы пометить нормальные
        значения как незнакомые.
        """
        return known_categories(preprocessor)

    def meta_learning(self) -> Path:
        """Анализ накопленных прогонов (Meta Learning, 7.b.iii).

        Читает манифесты всех обработанных батчей и записывает
        `reports/meta.json`: влияние настроек, влияние условий
        обучения, динамику метрик данных и моделей, выводы словами.

        Вызывается из `summary`, а не из обработки батча: анализ имеет
        смысл только по накопленной истории, а считать его на каждом
        батче значит платить O(n²) без выигрыша.
        """
        manifests = read_artifacts(self.config.metadata_dir, "run_manifest")
        payload = analyse_meta(
            manifests,
            metric=self.config.meta_metric,
            min_runs=self.config.meta_min_runs,
        )
        path = save_json(self.config.reports_dir / "meta.json", payload)
        logger.info(
            "Meta Learning: батчей %d, прогонов %d, выводов %d",
            payload.get("n_batches", 0), payload.get("n_runs", 0),
            len(payload.get("findings") or []),
        )
        return path

    def publish(self) -> Path:
        """Собрать только сайт, без отчёта и дашборда.

        Отдельный метод нужен режиму `-mode publish`: он зовётся
        отдельно от `summary` и не должен заодно переписывать отчёт.
        Обращается к тому же порту представлений, что и `summary`,
        иначе вывод снова стал бы частью контроллера (7.b.iv).
        """
        context = PipelineContext(config=self.config, state=self.state)
        results = self.views.render_all(context, only=["site"])
        outputs = {
            item.view: item.path for item in results if item.ok and item.path
        }
        site = outputs.get("site")
        if site is None:
            built = ", ".join(
                f"{item.view}: {item.error or 'ок'}" for item in results
            )
            raise PipelineError(f"Сайт не собран ({built})")
        return site

    def summary(self) -> tuple[Path, Path]:
        """Построить все артефакты вывода через слой представлений.

        Оркестратор не знает, какие представления зарегистрированы и в
        каком порядке: он отдаёт контекст и получает список результатов
        (7.b.iv). Поэтому добавление нового вида вывода — это одна
        регистрация, а не правка этого метода.

        Returns:
            (путь к текстовому отчёту, путь к дашборду) — те два
            файла, о которых сообщает CLI. Остальные лежат в
            `context.extras` и в каталоге отчётов.
        """
        context = PipelineContext(config=self.config, state=self.state)
        results = self.views.render_all(context)

        outputs = {
            item.view: item.path for item in results if item.ok and item.path
        }
        for item in results:
            if not item.ok:
                logger.warning("Вывод %s не построен: %s", item.view, item.error)

        report = outputs.get("text_report")
        dashboard = outputs.get("html_dashboard")
        if report is None or dashboard is None:
            built = ", ".join(
                f"{item.view}: {item.error or 'ок'}" for item in results
            )
            raise PipelineError(f"Не все артефакты вывода построены ({built})")
        return report, dashboard
