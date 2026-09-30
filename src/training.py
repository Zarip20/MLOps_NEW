"""Обучение и дообучение моделей.

Ключевое отличие от исходной реализации — порядок действий. Раньше
`train_models` обучалась на всём батче, и только потом вызывался
`train_test_split` на той же матрице, то есть метрики считались на
данных, участвовавших в обучении (D-2). Теперь разделение выполняется
первым, и валидационная часть не попадает в обучение никогда.

Схема соответствует заданию (`task.md` §4.1 и §5):

* текущий батч делится на train/val, 20 % откладывается под валидацию
  со стратификацией по метке;
* **дерево решений** переобучается с нуля на накопленных train-частях
  прошлых батчей плюс train-часть текущего;
* **нейросеть** дообучается `partial_fit` только на новой train-части —
  это и есть инкрементальное обучение (D-3);
* метрики считаются только на val-части.

Дисбаланс классов (доля положительного класса падает с 11 % до 2.6 %)
компенсируется весами образцов: у `MLPClassifier` нет параметра
`class_weight`, но `partial_fit` принимает `sample_weight`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import Config
from src.storage import TrainingStore
from src.utils import atomic_pickle_dump, safe_pickle_load

logger = logging.getLogger(__name__)

#: Конструкторы моделей, поддерживаемые конвейером.
#:
#: Набор намеренно разнородный по устойчивости — это и есть требование
#: 4.b.ii («несколько моделей с различной устойчивостью к входным данным»):
#:
#: * `lr`  — логистическая регрессия: почти не требует настройки, устойчива
#:   к нелинейным взаимодействиям отсутствию оных, даёт коэффициенты
#:   для интерпретации (5.b.i);
#: * `dt`  — дерево решений: ловит нелинейности и взаимодействия, но
#:   склонно переобучаться на дисбалансе;
#: * `rf`  — ансамбль деревьев: устойчивее одиночного дерева, медленнее;
#: * `mlp` — нейросеть: требует масштабирования, чувствительна к объёму
#:   данных, единственная из четырёх поддерживает дообучение `partial_fit`
#:   (4.b.i).
MODEL_CLASSES: dict[str, Any] = {
    "lr": LogisticRegression,
    "dt": DecisionTreeClassifier,
    "rf": RandomForestClassifier,
    "mlp": MLPClassifier,
}

#: Служебные ключи секции модели — не параметры конструктора.
SERVICE_KEYS = ("partial_fit", "epochs_per_batch", "use_store", "balancing")

#: Порядок обучения моделей в батче. Первой идёт самая дешёвая и устойчивая
#: модель, чтобы при обрыве батча в реестр уже попал хоть какой-то кандидат.
DEFAULT_ORDER = ("lr", "dt", "rf", "mlp")

METRIC_NAMES = ("precision", "recall", "f1", "roc_auc", "accuracy")


class InsufficientDataError(RuntimeError):
    """В батче не хватает данных для обучения или валидации."""


@dataclass
class TrainingOutcome:
    """Результат обучения на одном батче."""

    models: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    durations: dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0
    positive_rate_train: float = 0.0
    positive_rate_val: float = 0.0
    notes: list[str] = field(default_factory=list)
    preprocessor: Any = None
    train_part: pd.DataFrame | None = None
    #: Фактически применённые настройки по каждой модели. Нужны для
    #: Meta Learning (7.b.iii): без них по накопленным прогонам нельзя
    #: оценить влияние настроек, потому что неизвестно, чем именно
    #: обучалась та или иная версия.
    hyperparameters: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def best_model_name(self) -> str | None:
        """Модель с наибольшим f1 — по условию задания отбор идёт по f1."""
        if not self.metrics:
            return None
        return max(self.metrics, key=lambda name: self.metrics[name].get("f1", 0.0))


# ---------------------------------------------------------------------------
# Конструирование и загрузка
# ---------------------------------------------------------------------------


def model_order(config: Config) -> list[str]:
    """Порядок обработки моделей: сначала заданные в конфигурации."""
    configured = [name for name in DEFAULT_ORDER if name in config.models]
    extra = [name for name in config.models if name not in configured]
    return configured + extra


def constructor_args(config: Config, name: str) -> dict[str, Any]:
    """Гиперпараметры конструктора без служебных ключей."""
    params = dict(config.models.get(name, {}))
    for key in SERVICE_KEYS:
        params.pop(key, None)
    params.setdefault("random_state", config.run.get("seed", 42))
    return params


def build_models(config: Config) -> dict[str, Any]:
    """Создать не обученные модели по гиперпараметрам конфигурации."""
    return {
        name: MODEL_CLASSES[name](**constructor_args(config, name))
        for name in model_order(config)
    }


def load_models(config: Config) -> dict[str, Any] | None:
    """Загрузить ранее обученные модели; если их нет — `None`."""
    models: dict[str, Any] = {}
    for name in model_order(config):
        path = config.models_dir / f"{name}_latest.pkl"
        if path.is_file():
            try:
                models[name] = safe_pickle_load(path)
            except ValueError as error:
                logger.warning("%s", error)
    return models or None


def save_model(config: Config, model: Any, filename: str) -> None:
    """Сериализовать модель в каталог моделей (атомарно)."""
    path = atomic_pickle_dump(model, config.models_dir / filename)
    logger.info("Модель сохранена: %s (%s)", path.name, type(model).__name__)


# ---------------------------------------------------------------------------
# Разделение и веса
# ---------------------------------------------------------------------------


def split_batch(
    frame: pd.DataFrame,
    validation: dict[str, Any],
    min_positive: int = 50,
    target: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Разделить батч на train/val.

    Стратификация гарантирует, что в обеих частях сохранится исходная доля
    положительного класса: без неё батч с 3 % положительных дал бы валидацию
    почти без примеров класса 1 и f1, скачущий от батча к батчу.

    Args:
        frame: батч с признаками и целевой меткой.
        validation: параметры разбиения.
        min_positive: ниже этого числа примеров класса 1 добавляется
            предупреждение о неустойчивости метрик.
        target: имя целевой колонки. Если не задано, берётся последняя
            колонка. Явно указывать нужно всегда: в кадре конвейера
            последней может оказаться временная колонка, и её использование
            как метки молча дало бы «0 положительных примеров».

    Returns:
        (train, val, примечания).
    """
    notes: list[str] = []
    test_size = float(validation.get("test_size", 0.2))
    stratify_enabled = bool(validation.get("stratify", True))
    random_state = validation.get("random_state", 42)

    target = target or frame.columns[-1]
    if target not in frame.columns:
        raise InsufficientDataError(
            f"Целевая колонка {target!r} отсутствует в батче; доступны: {list(frame.columns)}"
        )

    positive_count = int((frame[target] == 1).sum())
    negative_count = int(len(frame) - positive_count)

    # Стратификация требует минимум по 2 представителя каждого класса.
    stratify: pd.Series | None = None
    if stratify_enabled and min(positive_count, negative_count) >= 2:
        stratify = frame[target]
    else:
        stratify = None
        notes.append(
            f"Стратификация отключена: в батче {positive_count} положительных "
            f"и {negative_count} отрицательных примеров"
        )

    if min(positive_count, negative_count) < min_positive:
        notes.append(
            f"Мало примеров класса 1 ({positive_count}); метрики будут неустойчивы"
        )

    train, val = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    return train, val, notes


def compute_sample_weight(y: np.ndarray, strategy: str = "balanced") -> np.ndarray:
    """Веса образцов для компенсации дисбаланса классов.

    `balanced` даёт каждому классу суммарный вес, пропорциональный обратной
    частоте. Другие стратегии: `none` — равные веса, `sqrt` — сглаженная
    компенсация (менее агрессивная, подходит при сильном дисбалансе).
    """
    if strategy == "none":
        return np.ones(len(y), dtype=float)

    counts = np.bincount(y.astype(int), minlength=2).astype(float)
    counts[counts == 0] = 1.0
    n_samples = len(y)
    n_classes = len(counts)

    if strategy == "sqrt":
        weights = np.sqrt(n_samples / (n_classes * counts[y.astype(int)]))
    else:  # balanced
        weights = n_samples / (n_classes * counts[y.astype(int)])

    return weights.astype(float)


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------


def train_batch(
    config: Config,
    frame: pd.DataFrame,
    store: TrainingStore | None,
    existing: dict[str, Any] | None,
    preprocessor: Any | None = None,
    numerical_cols: Sequence[str] | None = None,
    engineer: Any | None = None,
) -> TrainingOutcome:
    """Обучить модели на одном батче и посчитать метрики.

    Порядок шагов принципиален и нарушать его нельзя:

    1. **Разделение** train/val выполняется первым, на сырых признаках.
    2. **Препроцессор** обучается на train-части и только затем применяется
       к val-части. Если обучать его на всём батче, медианы импутера и
       параметры масштаба увидели бы валидационные данные — это вторая,
       более тонкая форма утечки, которой не было в исходном коде.
    3. **Обучение** выполняется на преобразованной train-части.
    4. **Оценка** — только на преобразованной val-части.

    Args:
        config: конфигурация конвейера.
        frame: батч с колонками-признаками и целевой меткой (последняя).
        store: накопительное хранилище train-частей прошлых батчей.
        existing: ранее обученные модели для дообучения.
        preprocessor: ранее обученный препроцессор. Если `None` — будет
            обучен заново на train-части текущего батча.
        numerical_cols: числовые признаки с учётом производных. Если не
            задан, берутся признаки из конфигурации.
        engineer: функция, применяемая к данным из накопительного
            хранилища перед отбором признаков. Хранилище намеренно
            содержит **исходные** колонки, а не производные: иначе смена
            набора инжиниринговых признаков сделала бы накопление
            несовместимым с текущей схемой обучения.

    Returns:
        `TrainingOutcome` с моделями, метриками, диагностикой и
        train-частью для накопительного хранилища.
    """
    from src.preprocessing import create_preprocessor

    # Производные признаки добавляются в числовые: они не требуют
    # кодирования и корректно проходят импутер и масштабирование.
    numerical = list(numerical_cols or config.numerical_cols)
    categorical = list(config.categorical_cols)
    feature_columns = numerical + categorical

    target = config.target_name
    outcome_notes: list[str] = []

    # 1. Разделение до обучения — валидация не должна видеть обучение (D-2).
    train_part, val_part, notes = split_batch(
        frame,
        config.validation,
        config.validation.get("min_positive_samples", 50),
        target=target,
    )
    outcome_notes.extend(notes)

    x_train_raw = train_part[feature_columns]
    y_train = train_part[target].to_numpy()
    x_val_raw = val_part[feature_columns]
    y_val = val_part[target].to_numpy()

    if len(np.unique(y_train)) < 2:
        raise InsufficientDataError(
            "В обучающей части батча присутствует только один класс — "
            "модель обучить не на чем"
        )

    # 2. Препроцессор обучается на train-части и применяется к обеим.
    if preprocessor is None:
        preprocessor = create_preprocessor(
            {
                **config.data["preprocessing"],
                "numerical_cols": numerical,
                "categorical_cols": categorical,
            }
        )
        x_train = preprocessor.fit_transform(x_train_raw)
        outcome_notes.append("препроцессор обучен на текущем батче")
    else:
        x_train = preprocessor.transform(x_train_raw)

    x_val = preprocessor.transform(x_val_raw)
    outcome_notes.append(f"признаков после предобработки: {x_train.shape[1]}")

    # 3. Данные для переобучения с нуля: накопленное + текущая train-часть.
    #    Метки расширяются вместе с признаками — иначе X и y разойдутся
    #    по длине, и scikit-learn отклонит обучение.
    store_cfg = config.get("training_store", {})
    window = store_cfg.get("window") if store_cfg.get("enabled") else None
    store_x, store_y = x_train, y_train
    store_rows = 0

    if store is not None and store_cfg.get("enabled") and len(store):
        history = store.load(window=window)
        if engineer is not None:
            # Накопленные данные приводятся к той же схеме признаков,
            # что и текущий батч, иначе матрицы не совпадут.
            history = engineer(history)
        if target in history.columns and not history.empty:
            store_rows = len(history)
            # Накопленные данные проходят через тот же препроцессор,
            # поэтому ширина матрицы гарантированно совпадает.
            combined = pd.concat(
                [history[feature_columns], x_train_raw], ignore_index=True
            )
            store_x = preprocessor.transform(combined)
            store_y = np.concatenate([history[target].to_numpy(), y_train])

    existing = existing or {}
    models: dict[str, Any] = {}
    durations: dict[str, float] = {}
    metrics: dict[str, dict[str, Any]] = {}
    hyperparameters: dict[str, dict[str, Any]] = {}

    for name in model_order(config):
        model_cfg = config.models.get(name, {})
        started = time.perf_counter()

        # Накопленная история используется только моделями с
        # `use_store: true`. Нейросеть дообучается на новых данных —
        # это и есть смысл partial_fit — и историю не переигрывает.
        use_store = bool(model_cfg.get("use_store", False))
        model_x = store_x if use_store else x_train
        model_y = store_y if use_store else y_train

        model, note, mode = _train_one(
            name, config, model_cfg, existing.get(name), model_x, model_y,
            store_rows if use_store else 0,
        )

        durations[name] = round(time.perf_counter() - started, 3)
        if model is None:
            continue
        if note:
            outcome_notes.append(note)

        models[name] = model
        metrics[name] = evaluate_model(name, model, x_val, y_val)
        hyperparameters[name] = _effective_hyperparameters(
            name, config, model_cfg, model, len(model_x),
            store_rows if use_store else 0, mode,
        )

    if not models:
        raise InsufficientDataError(
            "Ни одна модель не была обучена: " + "; ".join(outcome_notes)
        )

    return TrainingOutcome(
        models=models,
        metrics=metrics,
        durations=durations,
        n_train=len(train_part),
        n_val=len(val_part),
        positive_rate_train=float(np.mean(y_train)),
        positive_rate_val=float(np.mean(y_val)) if len(y_val) else 0.0,
        notes=outcome_notes,
        preprocessor=preprocessor,
        train_part=train_part,
        hyperparameters=hyperparameters,
    )


def _effective_hyperparameters(
    name: str,
    config: Config,
    model_cfg: dict[str, Any],
    model: Any,
    rows_used: int,
    store_rows: int,
    mode: str,
) -> dict[str, Any]:
    """Что фактически применено к одной модели на этом батче.

    Собирается из двух источников: параметров конструктора из
    конфигурации и состояния обученного объекта. Второй источник
    нужен для настроек, которые меняет не конфигурация, а сам
    scikit-learn: `max_iter` у `MLPClassifier` — это достигнутое число
    итераций, а не заданное, и подставлять заданное значило бы
    приписать модели несуществующее качество.

    Признак `training_mode` различает обучение с нуля и дообучение.
    Он передаётся из `_train_one`, а не вычисляется здесь заново:
    иначе записанный режим мог бы разойтись с тем, что на самом деле
    сделало обучение, а расхождение в метаданных заметно позже всего.

    У `mlp` режим меняется уже на первом батче: сначала обучение с
    нуля, затем дообучение. Это единственная настройка, которая во
    всём прогоне действительно варьируется, и потому только по ней
    влияние можно оценить статистически.
    """
    params = constructor_args(config, name)
    fitted = getattr(model, "get_params", lambda: {})() or {}

    result: dict[str, Any] = {}
    for key in sorted(params):
        value = params[key]
        if isinstance(value, (int, float, str, bool)) or value is None:
            result[key] = value

    # Достигнутые, а не заданные значения — только те, что у них есть.
    if "max_iter" in fitted:
        result["max_iter_reached"] = fitted.get("max_iter")
    if "n_iter_" in fitted and fitted.get("n_iter_") is not None:
        reached = fitted.get("n_iter_")
        try:
            result["n_iter_reached"] = int(np.max(reached))
        except (TypeError, ValueError):
            result["n_iter_reached"] = None

    result["training_mode"] = mode
    if "epochs_per_batch" in model_cfg:
        result["epochs_per_batch"] = int(model_cfg.get("epochs_per_batch", 1))
    result["balancing"] = _balancing_used(model_cfg, fitted)
    result["rows_used"] = int(rows_used)
    result["store_rows"] = int(store_rows)
    return result


def _balancing_used(model_cfg: dict[str, Any], fitted: dict[str, Any]) -> str:
    """Как именно в этом прогоне компенсировался дисбаланс классов.

    `class_weight` задан в конструкторе — дисбаланс учтён им, и веса
    образцов не применяются: одновременное использование обоих
    механизмов компенсировало бы дисбаланс дважды.
    """
    if fitted.get("class_weight") not in (None, "balanced"):
        return "class_weight"
    return f"sample_weight:{model_cfg.get('balancing', 'balanced')}"


def _train_one(
    name: str,
    config: Config,
    model_cfg: dict[str, Any],
    existing: Any,
    x_train: Any,
    y_train: np.ndarray,
    store_rows: int,
) -> tuple[Any, str, str]:
    """Обучить одну модель с учётом её специфики.

    Инкрементальное обучение поддерживает только `mlp` — у остальных
    моделей метода `partial_fit` нет, и они каждый раз обучаются заново.

    Returns:
        `(модель, примечание, режим)`, где режим — `partial_fit` или
        `scratch`. Он возвращается отсюда, а не вычисляется на стороне,
        именно чтобы запись в метаданных не могла разойтись с тем,
        что было сделано на самом деле.
    """
    use_partial = bool(model_cfg.get("partial_fit", False)) and hasattr(
        existing, "partial_fit"
    )

    if use_partial:
        model, note = _train_incremental(
            name, model_cfg, existing, x_train, y_train
        )
        return model, note, "partial_fit"

    model, note = _train_from_scratch(
        name, config, model_cfg, x_train, y_train, store_rows
    )
    return model, note, "scratch"


def _train_incremental(
    name: str,
    model_cfg: dict[str, Any],
    existing: Any,
    x_train: Any,
    y_train: np.ndarray,
) -> tuple[Any, str]:
    """Дообучить модель на новых данных без перезапуска оптимизации."""
    epochs = int(model_cfg.get("epochs_per_batch", 1))
    balancing = model_cfg.get("balancing", "balanced")
    matrix = np.asarray(x_train, dtype="float32")
    weights = compute_sample_weight(y_train, balancing)

    for _ in range(max(epochs, 1)):
        existing.partial_fit(
            matrix,
            y_train,
            # classes задаётся явно: на батче, где встретился только
            # один класс, sklearn принял бы набор классов из одной позиции
            # и сломался бы на следующем батче.
            classes=np.array([0, 1]),
            sample_weight=weights,
        )
    return existing, (
        f"{name}: дообучение на {len(matrix)} примерах, эпох: {epochs}"
    )


def _train_from_scratch(
    name: str,
    config: Config,
    model_cfg: dict[str, Any],
    x_train: Any,
    y_train: np.ndarray,
    store_rows: int,
) -> tuple[Any, str]:
    """Обучить модель с нуля.

    Дисбаланс компенсируется только один раз. Если у конструктора задан
    `class_weight`, веса передаются туда, а `sample_weight` не применяется.
    Одновременное использование обоих механизмов компенсирует дисбаланс
    дважды: в прогоне это стоило дереву решений 0.08 f1 (0.237 → 0.161) —
    модель предсказывала положительный класс заметно чаще, чем следовало.
    """
    matrix = np.asarray(x_train, dtype="float32")
    params = constructor_args(config, name)
    model = MODEL_CLASSES[name](**params)

    balancing = model_cfg.get("balancing", "balanced")
    uses_class_weight = params.get("class_weight") not in (None, "balanced")

    if uses_class_weight:
        # Дисбаланс уже учтён параметром конструктора.
        model.fit(matrix, y_train)
        weight_reason = "class_weight"
    else:
        weights = compute_sample_weight(y_train, balancing)
        model.fit(matrix, y_train, sample_weight=weights)
        weight_reason = f"sample_weight:{balancing}"

    note = (
        f"{name}: обучение с нуля на {len(matrix)} примерах "
        f"[дисбаланс: {weight_reason}]"
    )
    if store_rows:
        note += f" (из накопленного хранилища {store_rows} + текущие)"
    return model, note


def _train_dt(
    model_cfg: dict[str, Any],
    x_train: Any,
    y_train: np.ndarray,
    store_rows: int,
) -> tuple[Any, str]:
    """Обучить дерево решений с нуля на накопленных данных."""
    params = {key: value for key, value in model_cfg.items() if key not in SERVICE_KEYS}
    params.setdefault("random_state", 42)

    model = DecisionTreeClassifier(**params)
    model.fit(x_train, y_train)

    note = f"dt: обучение с нуля на {len(x_train)} примерах"
    if store_rows:
        note += f" (из накопленного хранилища {store_rows} + текущие)"
    return model, note


# ---------------------------------------------------------------------------
# Оценка
# ---------------------------------------------------------------------------


def evaluate_model(
    name: str,
    model: Any,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
) -> dict[str, Any]:
    """Посчитать метрики модели на валидационной части.

    Args:
        x_val: уже преобразованная валидационная матрица либо DataFrame.
    """
    predict = model.predict
    y_pred = predict(x_val)

    roc_auc: float | None = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(x_val)[:, 1]
            if len(np.unique(y_val)) > 1:
                roc_auc = float(roc_auc_score(y_val, probabilities))
        except (ValueError, IndexError, AttributeError):
            roc_auc = None

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()

    return {
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "predicted_positive_rate": float(np.mean(y_pred)),
        "actual_positive_rate": float(np.mean(y_val)) if len(y_val) else 0.0,
        "n_val": int(len(y_val)),
        "sklearn_version": sklearn.__version__,
    }
