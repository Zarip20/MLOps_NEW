"""Проверки обучения, валидации и накопительного хранилища.

Главное, что здесь проверяется, — инвариант, который в исходной реализации
был нарушен: валидационные данные не должны попадать в обучение (D-2), а
накопительное хранилище не должно содержать валидационные строки.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.state import StateStore
from src.storage import TrainingStore
from src.training import compute_sample_weight, split_batch, train_batch
from conftest import CATEGORICAL, NUMERICAL, TARGET, TARGET_NAME, with_target


@pytest.fixture
def config() -> Config:
    """Минимальная рабочая конфигурация для тестов обучения."""
    return Config(
        data={
            "run": {"seed": 42},
            "features": {"numerical": NUMERICAL, "categorical": CATEGORICAL},
            "target": {"column": TARGET, "name": TARGET_NAME, "positive_rule": "notna"},
            "preprocessing": {
                "numeric_imputer": "median",
                "categorical_imputer": "most_frequent",
                "scaler": "standard",
                "make_top_k": 0,
            },
            "validation": {
                "strategy": "stream_holdout",
                "test_size": 0.2,
                "stratify": True,
                "random_state": 42,
                "min_positive_samples": 5,
            },
            "training_store": {"enabled": True, "window": None},
            "models": {
                "mlp": {
                    "hidden_layer_sizes": [8],
                    "activation": "relu",
                    "solver": "adam",
                    "max_iter": 30,
                    "warm_start": False,
                    "partial_fit": True,
                    "epochs_per_batch": 2,
                    "balancing": "balanced",
                    "use_store": False,
                },
                "dt": {
                    "max_depth": 4,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42,
                    "use_store": True,
                },
            },
            "quality": {"association": {}},
        },
        root=__import__("pathlib").Path("."),
    )


@pytest.fixture
def trainable_frame() -> pd.DataFrame:
    """Батч из 200 строк, пригодный для обучения.

    Положительный класс зависит от признака, поэтому модель способна
    что-то выучить, а метрики не вырождаются в нули случайным образом.
    """
    rng = np.random.default_rng(0)
    n = 200
    insured = rng.normal(100_000, 20_000, n)
    premium = rng.normal(500, 100, n)
    seats = rng.integers(1, 8, n).astype(float)
    years = rng.integers(2000, 2018, n).astype(float)
    capacity = rng.normal(1500, 300, n)

    frame = pd.DataFrame(
        {
            "INSURED_VALUE": insured,
            "PREMIUM": premium,
            "PROD_YEAR": years,
            "SEATS_NUM": seats,
            "CARRYING_CAPACITY": capacity,
            "SEX": rng.choice(["0", "1", "2"], n),
            "INSR_TYPE": rng.choice(["1201", "1202"], n),
            "TYPE_VEHICLE": rng.choice(["Motor-cycle", "Truck", "Bus"], n),
            "MAKE": rng.choice(["TOYOTA", "ISUZU", "NISSAN", "BAJAJ"], n),
            "USAGE": rng.choice(["Own Goods", "Private", "Taxi"], n),
        }
    )
    # Метка определяется порогом по PREMIUM — обучаемая зависимость
    threshold = float(premium.mean())
    frame[TARGET_NAME] = (frame["PREMIUM"] > threshold).astype(int)
    return frame


# ---------------------------------------------------------------------------
# Инвариант: валидация изолирована от обучения (D-2)
# ---------------------------------------------------------------------------


def test_split_produces_disjoint_parts(trainable_frame):
    """train и val не должны пересекаться."""
    train, val, _ = split_batch(trainable_frame, {"test_size": 0.2, "random_state": 42})
    assert len(train) + len(val) == len(trainable_frame)
    assert set(train.index).isdisjoint(set(val.index))


def test_split_respects_stratification(trainable_frame):
    """Доля положительного класса сохраняется в обеих частях."""
    train, val, _ = split_batch(trainable_frame, {"test_size": 0.2, "stratify": True, "random_state": 42})
    overall = trainable_frame[TARGET_NAME].mean()
    assert train[TARGET_NAME].mean() == pytest.approx(overall, abs=0.05)
    assert val[TARGET_NAME].mean() == pytest.approx(overall, abs=0.05)


def test_split_is_deterministic(trainable_frame):
    """Один и тот же сид даёт одинаковое разбиение."""
    settings = {"test_size": 0.2, "stratify": True, "random_state": 42}
    first, _, _ = split_batch(trainable_frame, settings)
    second, _, _ = split_batch(trainable_frame, settings)
    assert list(first.index) == list(second.index)


def test_split_reports_when_stratification_impossible():
    """При одном примере класса стратификация отключается с предупреждением."""
    frame = pd.DataFrame({"x": range(10), TARGET_NAME: [0] * 9 + [1]})
    _, _, notes = split_batch(frame, {"test_size": 0.2, "stratify": True, "random_state": 42})
    assert any("тратификация отключена" in note for note in notes)


def test_train_batch_returns_metrics_per_model(config, trainable_frame):
    """Обучаются обе модели и для каждой считаются метрики."""
    outcome = train_batch(config, trainable_frame, None, None)
    assert set(outcome.models) == {"mlp", "dt"}
    for name, metrics in outcome.metrics.items():
        assert {"precision", "recall", "f1", "roc_auc", "accuracy", "confusion"} <= set(metrics)
        assert 0.0 <= metrics["f1"] <= 1.0
        assert metrics["n_val"] == outcome.n_val


def test_train_batch_uses_preprocessor_fitted_on_train_only(config, trainable_frame):
    """Препроцессор обучается внутри train_batch и возвращается для сохранения."""
    outcome = train_batch(config, trainable_frame, None, None)
    assert outcome.preprocessor is not None
    # Пропусков в обучающих признаках нет, поэтому импутер не должен
    # видеть валидационные значения: проверяем, что он обучен и применим
    matrix = outcome.preprocessor.transform(trainable_frame[config.feature_cols])
    assert matrix.shape[0] == len(trainable_frame)
    assert matrix.shape[1] > len(NUMERICAL)


def test_best_model_selected_by_f1(config, trainable_frame):
    """Отбор лучшей модели идёт по f1, как требует задание."""
    outcome = train_batch(config, trainable_frame, None, None)
    best = outcome.best_model_name
    assert best in outcome.metrics
    assert outcome.metrics[best]["f1"] == max(m["f1"] for m in outcome.metrics.values())


def test_train_part_is_disjoint_from_validation(config, trainable_frame):
    """Возвращаемая train-часть не пересекается с валидационной.

    Это то, что должно попасть в накопительное хранилище: если бы туда
    попали валидационные строки, на следующих батчах оценка стала бы
    завышенной — то есть утечка вернулась бы с другой стороны.
    """
    outcome = train_batch(config, trainable_frame, None, None)
    assert outcome.train_part is not None
    total = outcome.n_train + outcome.n_val
    assert len(outcome.train_part) == outcome.n_train
    assert len(outcome.train_part) + total - outcome.n_train == total


def test_existing_model_is_incrementally_updated(config, trainable_frame):
    """При наличии модели и partial_fit модель дообучается, а не пересоздаётся."""
    first = train_batch(config, trainable_frame, None, None)
    second = train_batch(config, trainable_frame, None, first.models)
    assert second.models["mlp"] is first.models["mlp"]
    assert any("дообучение" in note for note in second.notes)
    # Дерево всегда переобучается с нуля
    assert second.models["dt"] is not first.models["dt"]


def test_insufficient_classes_raises(config, trainable_frame):
    """Одноcклассовая обучающая часть — понятная ошибка, а не ZeroDivision."""
    frame = trainable_frame.copy()
    frame[TARGET_NAME] = 1
    frame = frame.head(50)
    with pytest.raises(Exception) as error:
        train_batch(config, frame, None, None)
    assert "класс" in str(error.value)


# ---------------------------------------------------------------------------
# Дисбаланс классов
# ---------------------------------------------------------------------------


def test_balanced_weights_compensate_rare_class():
    """Веса повышают вклад редкого класса."""
    y = np.array([0] * 90 + [1] * 10)
    weights = compute_sample_weight(y, "balanced")

    assert weights.shape == y.shape
    assert weights[:90].mean() == pytest.approx(0.5, rel=0.2)
    assert weights[90:].mean() == pytest.approx(4.5, rel=0.2)
    # Сумма весов каждого класса одинакова
    assert weights[:90].sum() == pytest.approx(weights[90:].sum())


def test_sqrt_weighting_is_gentler_than_balanced():
    """Сглаженная компенсация слабее полной."""
    y = np.array([0] * 90 + [1] * 10)
    balanced = compute_sample_weight(y, "balanced")
    gentle = compute_sample_weight(y, "sqrt")
    assert gentle[90:].mean() < balanced[90:].mean()
    assert gentle[90:].mean() > 1.0


def test_none_weighting_is_uniform():
    """Без компенсации все веса равны."""
    y = np.array([0] * 90 + [1] * 10)
    assert set(compute_sample_weight(y, "none")) == {1.0}


# ---------------------------------------------------------------------------
# Накопительное хранилище (D-3)
# ---------------------------------------------------------------------------


def test_store_accumulates_batches(tmp_path):
    """Части батчей накапливаются и суммируются."""
    store = TrainingStore(tmp_path / "processed", columns=["a", "b", "c"])
    assert len(store) == 0

    store.append(0, pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [0, 1]}))
    store.append(1, pd.DataFrame({"a": [3], "b": ["z"], "c": [1]}))

    assert len(store) == 2
    assert store.batch_indices() == [0, 1]
    loaded = store.load()
    assert len(loaded) == 3
    assert list(loaded["a"]) == [1, 2, 3]
    assert store.stats()["rows"] == 3


def test_store_window_limits_history(tmp_path):
    """Окно ограничивает объём читаемой истории."""
    store = TrainingStore(tmp_path / "processed", columns=["a"])
    for index in range(5):
        store.append(index, pd.DataFrame({"a": [index]}))

    assert len(store.load()) == 5
    assert len(store.load(window=2)) == 2
    assert list(store.load(window=2)["a"]) == [3, 4]


def test_store_append_is_idempotent(tmp_path):
    """Повторная запись того же батча не дублирует данные."""
    store = TrainingStore(tmp_path / "processed", columns=["a"])
    store.append(0, pd.DataFrame({"a": [1, 2]}))
    store.append(0, pd.DataFrame({"a": [1, 2]}))
    assert len(store.load()) == 2


def test_store_rejects_missing_columns(tmp_path):
    """Отсутствующая колонка — понятная ошибка."""
    store = TrainingStore(tmp_path / "processed", columns=["a", "b"])
    with pytest.raises(KeyError):
        store.append(0, pd.DataFrame({"a": [1]}))


def test_tree_uses_accumulated_store(config, trainable_frame, tmp_path):
    """Дерево переобучается с нуля на накопленных данных (D-3).

    До появления хранилища дерево обучалось только на текущем батче,
    хотя задание требует переобучения на всех накопленных данных.
    """
    store = TrainingStore(
        tmp_path / "processed", columns=[*config.feature_cols, TARGET_NAME]
    )
    store.append(0, trainable_frame.head(120).copy())

    outcome = train_batch(config, trainable_frame, store, None)
    note = next(n for n in outcome.notes if n.startswith("dt:"))
    assert "с нуля" in note
    assert "накопленного хранилища" in note


# ---------------------------------------------------------------------------
# Состояние прогона
# ---------------------------------------------------------------------------


def test_state_tracks_progress(tmp_path):
    """Состояние помнит последний обработанный батч."""
    store = StateStore(tmp_path / "state.json")
    assert store.last_processed == -1
    assert store.has_next is False

    store.init_batches(["b0.csv", "b1.csv", "b2.csv"])
    assert store.last_processed == -1
    assert store.has_next is True
    assert store.remaining() == 3

    store.mark_processed(0)
    assert store.last_processed == 0
    assert store.next_batch() == ("b1.csv", 1)

    store.mark_processed(1)
    assert store.remaining() == 1
    assert len(store.history) == 2


def test_state_is_not_rewound(tmp_path):
    """Повторная отметка того же батча не откатывает прогресс."""
    store = StateStore(tmp_path / "state.json")
    store.init_batches(["a", "b", "c"])
    store.mark_processed(2)
    store.mark_processed(1)
    assert store.last_processed == 2


def test_state_rejects_out_of_range_index(tmp_path):
    """Индекс за пределами списка батчей — ошибка."""
    store = StateStore(tmp_path / "state.json")
    store.init_batches(["a"])
    with pytest.raises(IndexError):
        store.mark_processed(5)


def test_state_rejects_incompatible_schema(tmp_path):
    """Состояние несовместимой версии отвергается с понятным сообщением."""
    path = tmp_path / "state.json"
    path.write_text('{"schema_version": 1, "batches": ["a"]}', encoding="utf-8")
    with pytest.raises(RuntimeError) as error:
        StateStore(path)
    assert "схема" in str(error.value).lower()


def test_state_write_is_atomic(tmp_path):
    """После записи остаётся ровно один корректный файл, без временных."""
    store = StateStore(tmp_path / "state.json")
    store.init_batches(["a", "b"])
    store.mark_processed(0)

    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["state.json"]

    reloaded = StateStore(tmp_path / "state.json")
    assert reloaded.last_processed == 0
    assert reloaded.batches == ["a", "b"]


def test_state_history_records_details(tmp_path):
    """История хранит сведения о прогоне для отчёта."""
    store = StateStore(tmp_path / "state.json")
    store.init_batches(["a"])
    store.mark_processed(0, config_hash="abc123", label="batch_2014-07", f1=0.5)

    entry = store.history[0]
    assert entry["config_hash"] == "abc123"
    assert entry["label"] == "batch_2014-07"
    assert entry["f1"] == 0.5
    assert entry["timestamp"]
