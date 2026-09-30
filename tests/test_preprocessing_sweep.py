"""Проверки перебора вариантов предобработки (3.b.i).

Перебор легко написать так, что он выдаёт «победителя» даже тогда, когда
победить не на чем. Поэтому проверяется именно честность: неудачные
варианты не выбрасываются молча, отсутствие результата распознаётся, а
применение перебора к прогону ограничено.
"""

from __future__ import annotations

import pytest

from src.config import Config
from src.preprocessing_sweep import (
    evaluate,
    pick,
    should_run,
    variants,
)


def make_config(tmp_path, declared=None, sweep=None) -> Config:
    data = {
        "paths": {
            "data_raw": "raw", "models": "m", "reports": "r",
            "metadata": "md", "state_file": "state.json",
            "rules_file": "rules.json",
        },
        "batching": {"size": "month", "time_column": "INSR_BEGIN",
                     "date_format": "%d-%b-%y"},
        "target": {"column": "CLAIM_PAID", "name": "HAS_CLAIM"},
        "validation": {"test_size": 0.3, "stratify": True, "random_state": 42},
        "preprocessing": {
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaler": "standard",
            "make_top_k": 0,
            "top_k_columns": ["MAKE"],
            "models": {},
        },
        "models": {"lr": {"max_iter": 200, "C": 1.0}},
    }
    if declared is not None:
        data["preprocessing"]["variants"] = declared
    if sweep is not None:
        data["preprocessing"]["sweep"] = sweep
    return Config(data=data, root=tmp_path)


def make_frame(rows: int = 400):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "INSURED_VALUE": rng.normal(100000, 20000, rows).round(),
        "PREMIUM": rng.normal(500, 80, rows).round(2),
        "CARRYING_CAPACITY": rng.normal(1500, 300, rows).round(),
        "MAKE": rng.choice(["ISUZU", "DAF", "AFRO", "BAJAJ"], rows),
        "USAGE": rng.choice(["Own Goods", "General Cartage"], rows),
    })
    frame["HAS_CLAIM"] = (rng.random(rows) < 0.1).astype(int)
    return frame


# ---------------------------------------------------------------------------
# Разбор вариантов
# ---------------------------------------------------------------------------


def test_no_variants_declared_means_base_only(tmp_path):
    """Без списка вариантов перебор не изобретает свою конфигурацию."""
    assert variants(make_config(tmp_path)) == [{}]


def test_variants_keep_declared_names(tmp_path):
    """Имена вариантов берутся из конфигурации, а не генерируются."""
    config = make_config(tmp_path, declared=[
        {"name": "median_standard"},
        {"name": "mean_robust", "numeric_imputer": "mean", "scaler": "robust"},
    ])
    assert [item["name"] for item in variants(config)] == [
        "median_standard", "mean_robust",
    ]


def test_unnamed_variant_gets_a_readable_name(tmp_path):
    """Без имени вариант всё равки называется читаемо."""
    config = make_config(tmp_path, declared=[{"scaler": "robust"}])
    name = variants(config)[0]["name"]

    assert "scaler" in name and "robust" in name


def test_unknown_imputer_is_rejected(tmp_path):
    """Неизвестная стратегия импутации отвергается сразу."""
    config = make_config(tmp_path, declared=[{"numeric_imputer": "выдумка"}])
    with pytest.raises(ValueError) as error:
        variants(config)
    assert "numeric_imputer" in str(error.value)


def test_unknown_scaler_is_rejected(tmp_path):
    """Неизвестный масштаб отвергается сразу."""
    config = make_config(tmp_path, declared=[{"scaler": "выдумка"}])
    with pytest.raises(ValueError) as error:
        variants(config)
    assert "scaler" in str(error.value)


def test_negative_top_k_is_rejected(tmp_path):
    """Отрицательное top-K бессмысленно и должно быть ошибкой."""
    config = make_config(tmp_path, declared=[{"make_top_k": -5}])
    with pytest.raises(ValueError) as error:
        variants(config)
    assert "top_k" in str(error.value)


# ---------------------------------------------------------------------------
# Перебор
# ---------------------------------------------------------------------------


def split(frame):
    train = frame.iloc[:300]
    return frame, train, frame.iloc[300:]


def test_sweep_returns_one_result_per_variant(tmp_path):
    """Каждому варианту соответствует ровно один результат."""
    config = make_config(tmp_path, declared=[
        {"name": "median_standard"},
        {"name": "mean_standard", "numeric_imputer": "mean"},
        {"name": "robust", "scaler": "robust"},
    ])
    frame, train, val = split(make_frame())
    results = evaluate(
        config, frame, train, val, "HAS_CLAIM",
        ["INSURED_VALUE", "PREMIUM", "CARRYING_CAPACITY"],
        ["MAKE", "USAGE"],
        model_params={"max_iter": 100},
    )

    assert len(results) == 3
    assert all(item["status"] == "ok" for item in results)
    assert all(item["f1"] is not None for item in results)


def test_top_k_reduces_feature_count(tmp_path):
    """Ограничение top-K действительно сокращает число признаков."""
    config = make_config(tmp_path, declared=[
        {"name": "все категории", "make_top_k": 0},
        {"name": "top-2", "make_top_k": 2},
    ])
    frame, train, val = split(make_frame())
    results = evaluate(
        config, frame, train, val, "HAS_CLAIM",
        ["INSURED_VALUE", "PREMIUM", "CARRYING_CAPACITY"],
        ["MAKE", "USAGE"],
        model_params={"max_iter": 100},
    )
    by_name = {item["name"]: item for item in results}

    assert by_name["top-2"]["n_features"] < by_name["все категории"]["n_features"]


def test_failing_variant_does_not_stop_the_sweep(tmp_path):
    """Сбой при построении не оборвал бы перебор, а уронил бы его целиком.

    Ошибка внутри одного варианта перехватывается, и остальные остаются
    сравнимыми. Список признаков здесь короче данных наоборот — так
    выглядит рассинхрон схемы, и перебор обязан выжить на нём, а не
    упасть вместе со всем прогоном.
    """
    config = make_config(tmp_path, declared=[
        {"name": "рабочий", "scaler": "standard"},
        {"name": "mean_standard", "numeric_imputer": "mean"},
    ])
    frame, train, val = split(make_frame())
    # В списке числовых признаков есть колонка, которой в кадре нет.
    results = evaluate(
        config, frame, train, val, "HAS_CLAIM",
        ["INSURED_VALUE", "PREMIUM", "ОТСУТСТВУЕТ"],
        ["MAKE", "USAGE"],
        model_params={"max_iter": 100},
    )

    assert len(results) == 2
    assert all(item["status"] == "failed" for item in results)
    assert all(item["error"] for item in results)
    # И перебор честно признаёт, что выбирать не из чего.
    assert pick(results)["status"] == "none"


# ---------------------------------------------------------------------------
# Выбор победителя
# ---------------------------------------------------------------------------


def test_pick_chooses_the_best(tmp_path):
    """Побеждает вариант с наибольшей метрикой."""
    results = [
        {"name": "a", "status": "ok", "f1": 0.2},
        {"name": "b", "status": "ok", "f1": 0.5},
        {"name": "c", "status": "ok", "f1": 0.3},
    ]
    assert pick(results)["name"] == "b"


def test_pick_ignores_failed_variants():
    """Сломанный вариант не может стать победителем."""
    results = [
        {"name": "сломанный", "status": "failed", "f1": 0.99},
        {"name": "рабочий", "status": "ok", "f1": 0.1},
    ]
    assert pick(results)["name"] == "рабочий"


def test_pick_reports_ties():
    """Ничья объявляется, а не разрешается молча."""
    results = [
        {"name": "a", "status": "ok", "f1": 0.5},
        {"name": "b", "status": "ok", "f1": 0.5},
    ]
    assert pick(results)["n_tied"] == 2


def test_pick_says_when_nothing_worked():
    """Если не построилось ничего, это признаётся, а не выдаётся за победу."""
    results = [{"name": "a", "status": "failed", "error": "boom"}]
    winner = pick(results)

    assert winner["status"] == "none"
    assert "не построился" in winner["reason"]


def test_pick_without_metric_is_not_a_winner():
    """Вариант без значения метрики не может победить."""
    results = [
        {"name": "a", "status": "ok", "f1": None},
        {"name": "b", "status": "ok", "f1": 0.3},
    ]
    assert pick(results)["name"] == "b"


# ---------------------------------------------------------------------------
# Когда перебор запускается
# ---------------------------------------------------------------------------


def test_sweep_runs_on_first_batch_by_default(tmp_path):
    """По умолчанию перебор идёт на первом батче и больше нигде."""
    config = make_config(tmp_path, sweep={"enabled": True})
    assert should_run(config, 0) is True
    for index in (1, 5, 20, 47):
        assert should_run(config, index) is False


def test_sweep_can_be_disabled(tmp_path):
    """Выключенный перебор не запускается нигде."""
    config = make_config(tmp_path, sweep={"enabled": False})
    assert should_run(config, 0) is False


def test_sweep_can_repeat_every_n_batches(tmp_path):
    """Периодический перебор настроен и считается от первого батча."""
    config = make_config(
        tmp_path, sweep={"enabled": True, "first_batch": 0, "every_n_batches": 10}
    )
    assert should_run(config, 0) is True
    assert should_run(config, 10) is True
    assert should_run(config, 20) is True
    assert should_run(config, 7) is False


def test_sweep_starts_at_configured_batch(tmp_path):
    """Первый батч перебора задаётся конфигурацией."""
    config = make_config(
        tmp_path, sweep={"enabled": True, "first_batch": 5, "every_n_batches": 0}
    )
    assert should_run(config, 0) is False
    assert should_run(config, 5) is True
    assert should_run(config, 6) is False
