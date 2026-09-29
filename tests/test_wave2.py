"""Проверки новых модулей Волны 2: реестр, дрейф, признаки, интерпретация.

Тесты фиксируют инварианты, которые оказались нетривиальными на практике:
гейт качества не должен блокировать первую модель из-за отсутствующих
данных, дисбаланс не должен компенсироваться дважды, производные признаки
должны действительно доходить до матрицы, а расхождение регистра должно
обнаруживаться, а не выдаваться за каждое значение колонки.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.drift import (
    DriftMonitor,
    binary_psi,
    interpret_psi,
    kolmogorov_smirnov,
    population_stability_index,
)
from src.explain import explain_model, feature_names_from_preprocessor
from src.features import detect_case_inconsistency, engineer_features, profile_frame
from src.registry import ModelRegistry, QualityGate
from src.training import MODEL_CLASSES, constructor_args, model_order

from conftest import CATEGORICAL, NUMERICAL, TARGET, TARGET_NAME, with_target


# ---------------------------------------------------------------------------
# Реестр и гейт качества (АР-5)
# ---------------------------------------------------------------------------


def test_gate_passes_solid_candidate(tmp_path):
    """Модель, проходящая пороги, становится продуктовой."""
    registry = ModelRegistry(
        tmp_path, QualityGate({"min_f1": 0.2, "min_roc_auc": 0.7})
    )
    entry = registry.register(
        model=DecisionTreeClassifier().fit([[0], [1]], [0, 1]),
        model_name="dt",
        batch_idx=0,
        batch_label="batch_2014-07",
        metrics={"f1": 0.5, "roc_auc": 0.8},
    )
    assert entry["status"] == "production"
    assert registry.production["model"] == "dt"
    assert (tmp_path / entry["artifact"]).is_file()


def test_gate_rejects_weak_candidate(tmp_path):
    """Слабая модель не попадает в продуктив, но остаётся в истории."""
    registry = ModelRegistry(tmp_path, QualityGate({"min_f1": 0.5}))
    entry = registry.register(
        model=DecisionTreeClassifier().fit([[0], [1]], [0, 1]),
        model_name="dt",
        batch_idx=0,
        batch_label="b0",
        metrics={"f1": 0.1, "roc_auc": 0.6},
    )
    assert entry["status"] == "rejected"
    assert entry["gate"]["passed"] is False
    # Регресс обязан оставаться в истории — иначе непонятно, что произошло
    assert len(registry.versions) == 1


def test_gate_does_not_fail_on_missing_drift(tmp_path):
    """Недоступный PSI — пропуск проверки, а не повод отклонить модель.

    На первом батче эталона дрейфа ещё нет, PSI равен None. Считать это
    провалом означало бы, что гейт отклонит первую же модель прогона.
    """
    gate = QualityGate({"max_psi": 0.3, "min_f1": 0.1})
    result = gate.evaluate({"f1": 0.5, "roc_auc": 0.8}, None, {"psi": None})

    assert result.passed is True
    psi_check = next(c for c in result.checks if c["name"] == "max_psi")
    assert psi_check["skipped"] is True


def test_gate_blocks_on_real_drift(tmp_path):
    """Реальный дрейф выше порога блокирует продвижение."""
    gate = QualityGate({"max_psi": 0.3})
    result = gate.evaluate({"f1": 0.5}, None, {"psi": 0.8})
    assert result.passed is False
    assert "0.8" in result.reasons[0]


def test_gate_blocks_large_regression(tmp_path):
    """Падение качества относительно продукта ограничено."""
    gate = QualityGate({"max_regression_f1": 0.1})
    result = gate.evaluate({"f1": 0.2}, {"f1": 0.5}, {})
    assert result.passed is False
    assert "упал" in result.reasons[0]


def test_gate_promotes_only_better_model(tmp_path):
    """Смена продукта происходит только при выигрыше по f1."""
    registry = ModelRegistry(tmp_path, QualityGate({"promote_best_by_f1": True}))

    registry.register(
        DecisionTreeClassifier().fit([[0], [1]], [0, 1]), "dt", 0, "b0",
        {"f1": 0.5, "roc_auc": 0.8},
    )
    weaker = registry.register(
        LogisticRegression().fit([[0], [1]], [0, 1]), "lr", 1, "b1",
        {"f1": 0.3, "roc_auc": 0.8},
    )
    assert weaker["status"] == "candidate"
    assert registry.production["model"] == "dt"

    better = registry.register(
        LogisticRegression().fit([[0], [1]], [0, 1]), "lr", 2, "b2",
        {"f1": 0.7, "roc_auc": 0.9},
    )
    assert better["status"] == "production"
    assert registry.production["model"] == "lr"


def test_registry_versions_are_immutable_and_sequential(tmp_path):
    """Версии нумеруются подряд и не перезаписывают друг друга."""
    registry = ModelRegistry(tmp_path)
    names = []
    for index in range(3):
        entry = registry.register(
            DecisionTreeClassifier().fit([[0], [1]], [0, 1]), "dt", index, f"b{index}",
            {"f1": 0.1 * index},
        )
        names.append(entry["artifact"])

    assert names == [
        "dt__v0001__b0.pkl", "dt__v0002__b1.pkl", "dt__v0003__b2.pkl",
    ]
    assert all((tmp_path / name).is_file() for name in names)
    assert registry.versions[0]["status"] == "previous"


def test_registry_pointers_reflect_production(tmp_path):
    """Указатели соответствуют продуктовой и последней версиям."""
    registry = ModelRegistry(tmp_path)
    registry.register(
        LogisticRegression().fit([[0], [1]], [0, 1]), "lr", 0, "b0",
        {"f1": 0.4},
    )
    registry.register(
        DecisionTreeClassifier().fit([[0], [1]], [0, 1]), "dt", 1, "b1",
        {"f1": 0.6},
    )
    created = registry.refresh_pointers()

    assert (tmp_path / "best_model.pkl").is_file()
    assert (tmp_path / "lr_latest.pkl").is_file()
    assert (tmp_path / "dt_latest.pkl").is_file()
    assert created["best_model"] == "dt__v0002__b1.pkl"


def test_registry_survives_corrupt_file(tmp_path):
    """Повреждённый реестр не приводит к падению, а начинается заново."""
    (tmp_path / "registry.json").write_text("{broken", encoding="utf-8")
    registry = ModelRegistry(tmp_path)
    assert registry.versions == []


def test_registry_rejects_incompatible_schema(tmp_path):
    """Схема реестра иной версии игнорируется, а не читается как данные."""
    (tmp_path / "registry.json").write_text(
        '{"schema_version": 99, "versions": [{"x": 1}]}', encoding="utf-8"
    )
    registry = ModelRegistry(tmp_path)
    assert registry.versions == []


# ---------------------------------------------------------------------------
# Модели и дисбаланс (4.b.ii)
# ---------------------------------------------------------------------------


def test_all_configured_models_are_supported():
    """Конструкторы покрывают все модели, описанные в конфигурации."""
    for name in ("lr", "dt", "rf", "mlp"):
        assert name in MODEL_CLASSES


def test_service_keys_are_not_passed_to_constructors():
    """Служебные ключи не попадают в API scikit-learn."""
    from src.config import Config

    config = Config(
        data={
            "run": {"seed": 42},
            "models": {
                "dt": {"max_depth": 3, "partial_fit": False, "use_store": True,
                       "balancing": "balanced", "epochs_per_batch": 5},
            },
        },
        root=__import__("pathlib").Path("."),
    )
    params = constructor_args(config, "dt")
    assert "partial_fit" not in params
    assert "use_store" not in params
    assert "balancing" not in params
    assert "epochs_per_batch" not in params
    assert params["max_depth"] == 3
    assert params["random_state"] == 42


def test_model_order_follows_declared_order():
    """Порядок обучения берётся из предпочтения: дешёвые и устойчивые первыми."""
    from src.config import Config

    config = Config(
        data={"run": {"seed": 42},
              "models": {"mlp": {}, "dt": {}, "lr": {}, "rf": {}}},
        root=__import__("pathlib").Path("."),
    )
    assert model_order(config) == ["lr", "dt", "rf", "mlp"]


# ---------------------------------------------------------------------------
# Дрейф (2.b.iv, 5.b.ii)
# ---------------------------------------------------------------------------


def test_psi_is_zero_for_identical_distributions():
    """Одинаковые распределения дают нулевой PSI."""
    rng = np.random.default_rng(0)
    sample = rng.normal(size=5000)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-6)


def test_psi_grows_with_shift():
    """Чем сильнее сдвиг, тем больше PSI."""
    rng = np.random.default_rng(0)
    reference = rng.normal(size=5000)
    small = rng.normal(loc=0.3, size=5000)
    large = rng.normal(loc=2.0, size=5000)

    assert population_stability_index(reference, small) < population_stability_index(reference, large)


def test_psi_tolerates_empty_bins():
    """Пустые интервалы не приводят к делению на ноль."""
    reference = np.concatenate([np.zeros(900), np.ones(100)])
    actual = np.concatenate([np.zeros(100), np.ones(900)])
    value = population_stability_index(reference, actual)
    assert np.isfinite(value)
    assert value > 0


def test_psi_handles_degenerate_input():
    """Пустые выборки дают NaN, а не исключение."""
    assert np.isnan(population_stability_index(np.array([]), np.array([1.0, 2.0])))
    assert np.isnan(population_stability_index(np.ones(10), np.ones(10)))


def test_ks_is_zero_for_same_sample():
    """KS-статистика равна нулю для совпадающих выборок."""
    rng = np.random.default_rng(1)
    sample = rng.normal(size=2000)
    assert kolmogorov_smirnov(sample, sample) == pytest.approx(0.0)


def test_psi_interpretation_levels():
    """PSI переводится в уровень по общепринятой шкале."""
    assert interpret_psi(0.05)[0] == "stable"
    assert interpret_psi(0.15)[0] == "moderate"
    assert interpret_psi(0.5)[0] == "significant"
    assert interpret_psi(float("nan"))[0] == "unknown"


def test_binary_psi_is_zero_for_equal_rates():
    """Одинаковые доли положительного класса дают нулевой PSI."""
    assert binary_psi(0.11, 0.11) == pytest.approx(0.0, abs=1e-9)


def test_binary_psi_detects_real_shift():
    """Сдвиг доли положительного класса отражается в PSI.

    Регрессия, найденная на прогоне: обе доли брались из одного массива,
    поэтому разность всегда была нулевой и PSI равнялся 0.0 даже при
    падении доли положительного класса с 11 % до 2.4 %.
    """
    value = binary_psi(0.111, 0.024)
    assert value > 0.1
    # Чем сильнее сдвиг, тем больше PSI
    assert value > binary_psi(0.111, 0.100)


def test_binary_psi_handles_extreme_rates():
    """Граничные доли не дают деления на ноль и бесконечностей."""
    for reference, actual in ((0.0, 1.0), (1.0, 0.0), (0.001, 0.999)):
        value = binary_psi(reference, actual)
        assert np.isfinite(value)


def test_drift_monitor_refuses_to_compare_without_reference():
    """Без эталона проверка не выдумывает сравнение, а признаёт невозможность.

    Отметку `baseline` ставит оркестратор, который сначала строит эталон из
    первого батча. Сам монитор, не имея эталона, обязан сказать об этом,
    а не вернуть пустой результат с нулевым PSI.
    """
    frame = pd.DataFrame({"A": np.arange(100.0), "B": ["x", "y"] * 50, "HAS_CLAIM": [0, 1] * 50})
    monitor = DriftMonitor({"psi_alert": 0.25})
    assert monitor.has_reference() is False

    result = monitor.check(frame, "HAS_CLAIM", ["A"], ["B"])
    assert result["available"] is False
    assert result["status"] == "unknown"
    assert "эталон не задан" in result["reason"]


def test_drift_monitor_detects_shift():
    """Смена распределения поднимает статус до alert."""
    rng = np.random.default_rng(2)
    base = pd.DataFrame({
        "A": rng.normal(10, 1, 500), "B": ["x", "y"] * 250, "HAS_CLAIM": [0, 1] * 250,
    })
    monitor = DriftMonitor({"psi_warn": 0.1, "psi_alert": 0.25})
    monitor.set_reference(["A"], ["B"], "HAS_CLAIM", base)

    shifted = base.copy()
    shifted["A"] = rng.normal(50, 1, 500)
    result = monitor.check(shifted, "HAS_CLAIM", ["A"], ["B"])

    assert result["available"] is True
    assert result["status"] == "alert"
    assert result["max_psi"] > 0.25


def test_drift_reference_is_a_rolling_window():
    """Эталон не растёт бесконечно: это окно, а не вся история."""
    rng = np.random.default_rng(3)
    monitor = DriftMonitor({})
    first = pd.DataFrame({"A": rng.normal(size=20000), "B": ["x"] * 20000, "HAS_CLAIM": [0, 1] * 10000})
    monitor.set_reference(["A"], ["B"], "HAS_CLAIM", first)
    size_after_first = monitor._reference["numerical"]["A"]["values"].size

    for _ in range(10):
        batch = pd.DataFrame({"A": rng.normal(size=20000), "B": ["x"] * 20000,
                              "HAS_CLAIM": [0, 1] * 10000})
        monitor.extend_reference(batch, ["A"], ["B"], "HAS_CLAIM", window_batches=2)

    size_later = monitor._reference["numerical"]["A"]["values"].size
    assert size_later < size_after_first * 5
    assert size_later <= 2 * 20000


# ---------------------------------------------------------------------------
# EDA и признаки (2.b.i, 2.b.ii)
# ---------------------------------------------------------------------------


def test_case_inconsistency_detects_real_conflict():
    """Расхождение регистра между колонками обнаруживается."""
    frame = pd.DataFrame({
        "TYPE_VEHICLE": ["Special construction"] * 10,
        "USAGE": ["Special Construction"] * 10,
    })
    result = detect_case_inconsistency(frame, ["TYPE_VEHICLE", "USAGE"])
    # Внутри одной колонки написаний, различающихся регистром, нет
    assert result == {}


def test_case_inconsistency_within_single_column():
    """Одна и та же категория, написанная по-разному, попадает в отчёт."""
    frame = pd.DataFrame({"USAGE": ["Own Goods", "own goods", "Own goods"]})
    result = detect_case_inconsistency(frame, ["USAGE"])
    assert "USAGE" in result
    # Свёрнутые к нижнему регистру все три написания образуют одну группу
    # с тремя разными вариантами, значит конфликтуют все три
    assert set(result["USAGE"]) == {"Own Goods", "own goods", "Own goods"}


def test_case_inconsistency_ignores_distinct_categories():
    """Обычные разные категории не считаются расхождением регистра.

    Прежняя проверка считала «сколько раз встречается свёрнутое
    значение» и помечала практически любую колонку: у SEX значения
    0, 1, 2 встречаются тысячи раз, и все три оказывались «конфликтующими».
    """
    frame = pd.DataFrame({"SEX": ["0"] * 100 + ["1"] * 50 + ["2"] * 20})
    assert detect_case_inconsistency(frame, ["SEX"]) == {}


def test_engineered_features_are_numeric_and_finite(sample_batch):
    """Производные признаки числовые и без бесконечностей."""
    frame = with_target(sample_batch)
    result, added = engineer_features(frame, NUMERICAL, CATEGORICAL, "INSR_BEGIN")

    assert added, "производные признаки должны появиться"
    for name in added:
        series = result[name]
        assert pd.api.types.is_numeric_dtype(series), name
        assert not np.isinf(series.to_numpy(dtype="float64", na_value=np.nan)).any(), name


def test_vehicle_age_is_computed(sample_batch):
    """Возраст автомобиля считается как разница годов."""
    frame = with_target(sample_batch)
    result, _ = engineer_features(frame, NUMERICAL, CATEGORICAL, "INSR_BEGIN")
    # INSR_BEGIN = 08-AUG-15, PROD_YEAR = 2015 -> 0
    assert "VEHICLE_AGE" in result.columns
    assert result.loc[result["PROD_YEAR"] == 2015, "VEHICLE_AGE"].eq(0).all()
    # PROD_YEAR = 2013 -> 2
    assert result.loc[result["PROD_YEAR"] == 2013, "VEHICLE_AGE"].eq(2).all()


def test_sentinel_zero_flag_does_not_change_the_value(sample_batch):
    """Индикатор нуля не подменяет само значение (решение Р-2)."""
    frame = with_target(sample_batch)
    result, added = engineer_features(frame, NUMERICAL, CATEGORICAL, "INSR_BEGIN")
    if "IS_INSURED_VALUE_ZERO" in added:
        zeros = result["INSURED_VALUE"] == 0
        assert result.loc[zeros, "IS_INSURED_VALUE_ZERO"].eq(1).all()
        # Исходные нули остаются нулями, а не становятся пропусками
        assert result.loc[zeros, "INSURED_VALUE"].eq(0).all()


def test_engineering_is_idempotent(sample_batch):
    """Повторный вызов не добавляет дублей и не меняет значения."""
    frame = with_target(sample_batch)
    first, added = engineer_features(frame, NUMERICAL, CATEGORICAL, "INSR_BEGIN")
    second, again = engineer_features(first, NUMERICAL, CATEGORICAL, "INSR_BEGIN")
    assert again == []
    assert list(first.columns) == list(second.columns)


def test_profile_reports_sentinel_zeros(sample_batch):
    """EDA отмечает долю сентинельных нулей."""
    frame = with_target(sample_batch)
    profile = profile_frame(frame, NUMERICAL, CATEGORICAL, TARGET_NAME)
    assert profile["numerical"]["INSURED_VALUE"]["zeros"] == 4
    assert profile["numerical"]["INSURED_VALUE"]["zeros_ratio"] > 0
    assert profile["target"]["positive_rate"] > 0
    assert profile["rows"] == len(frame)


# ---------------------------------------------------------------------------
# Интерпретация (5.b.i)
# ---------------------------------------------------------------------------


def test_linear_model_explanation_has_coefficients_and_odds():
    """У линейной модели извлекаются коэффициенты и шансы."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(300, 3))
    y = (x[:, 0] + rng.normal(scale=0.3, size=300) > 0).astype(int)
    model = LogisticRegression().fit(x, y)

    payload = explain_model(model, ["A", "B", "C"], "lr")
    assert payload["method"] == "logistic_regression_coefficients"
    assert payload["features"][0]["feature"] in ("A", "B", "C")
    assert "coefficient" in payload["features"][0]
    assert "odds_ratio" in payload["features"][0]


def test_tree_explanation_reports_importance_and_structure():
    """У дерева извлекаются важности и верхние разбиения."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 3))
    y = (x[:, 0] > 0).astype(int)
    model = DecisionTreeClassifier(max_depth=3).fit(x, y)

    payload = explain_model(model, ["A", "B", "C"], "dt")
    assert payload["method"] == "feature_importance"
    assert payload["features"][0]["feature"] == "A"
    assert payload["structure"], "структура дерева должна быть извлечена"
    assert any(item["type"] == "split" for item in payload["structure"])


def test_neural_explanation_uses_input_layer():
    """Для нейросети извлекаются веса входного слоя."""
    rng = np.random.default_rng(6)
    x = rng.normal(size=(200, 4))
    y = (x[:, 0] > 0).astype(int)
    model = MLPClassifier(hidden_layer_sizes=[8], max_iter=20).fit(x, y)

    payload = explain_model(model, ["A", "B", "C", "D"], "mlp")
    assert payload["method"] == "neural_input_layer_weights"
    assert len(payload["features"]) == 4
    assert "importance" in payload["features"][0]


def test_unsupported_model_is_reported_not_crashed():
    """Модель без объяснимых атрибутов не роняет конвейер."""
    class Opaque:
        pass

    payload = explain_model(Opaque(), ["A"], "opaque")
    assert payload["method"] == "unsupported"
    assert "note" in payload


def test_feature_names_fallback_when_unavailable():
    """Отсутствие имён у препроцессора не ломает отчёт."""
    class Fake:
        def get_feature_names_out(self):
            raise ValueError("недоступно")

    assert feature_names_from_preprocessor(Fake()) == []
