"""Проверки ассоциативных правил и служебных функций.

Ассоциативные правила проверяются на том, что отбор идёт по полезности
правила, а не по уверенности: в baseline-прогоне отбор по confidence дал
пять правил с confidence = 1.0, и все пять — в сторону мажоритного класса.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from src.association import (
    generate_association_rules,
    load_association_rules,
    rule_statistics,
)
from src.utils import (
    atomic_pickle_dump,
    load_json,
    safe_pickle_load,
    to_json_serializable,
)


@pytest.fixture
def transactions() -> pd.DataFrame:
    """Батч, в котором содержательные правила очевидны.

    Марка `ISUZU` почти всегда означает грузовик, а `BAJAJ` — мотоцикл.
    Связь строится явно, а не через независимые розыгрыши: иначе признаки
    оказываются независимыми, и ассоциативных правил не находится вовсе.
    """
    rng = np.random.default_rng(7)
    n = 2000

    make = rng.choice(["ISUZU", "BAJAJ", "TOYOTA"], n, p=[0.45, 0.30, 0.25])
    vehicle = np.empty(n, dtype=object)
    noise = rng.random(n)

    for index, brand in enumerate(make):
        if brand == "ISUZU":
            vehicle[index] = "Truck" if noise[index] < 0.85 else "Bus"
        elif brand == "BAJAJ":
            vehicle[index] = "Motor-cycle" if noise[index] < 0.90 else "Bus"
        else:
            vehicle[index] = (
                "Bus" if noise[index] < 0.50
                else "Motor-cycle" if noise[index] < 0.80
                else "Truck"
            )

    return pd.DataFrame(
        {
            "MAKE": make,
            "TYPE_VEHICLE": vehicle,
            # 1202 — мажоритный класс (70 %): правила в его сторону
            # не должны попадать в выдачу
            "INSR_TYPE": rng.choice(["1201", "1202"], n, p=[0.3, 0.7]),
        }
    )


def test_rules_are_structured_not_strings(tmp_path, transactions):
    """Правила хранятся парами колонка/значение, а не разбираются из строк."""
    rules = generate_association_rules(
        transactions, ["MAKE", "TYPE_VEHICLE", "INSR_TYPE"], tmp_path / "rules.json",
        min_support=0.01, min_confidence=0.5, min_lift=1.5, max_len=2, n_rules=5,
    )
    assert rules
    for rule in rules:
        assert isinstance(rule["antecedents"], list)
        for item in rule["antecedents"] + rule["consequents"]:
            assert set(item) == {"column", "value"}
        assert isinstance(rule["label"], str)


def test_majority_consequent_is_excluded(tmp_path, transactions):
    """Следствие, совпадающее с мажоритным значением, отбрасывается.

    Именно на этом ломался исходный отбор: INSR_TYPE = 1202 занимает
    70 % строк, и правила в его сторону всегда имели confidence = 1.0.
    """
    rules = generate_association_rules(
        transactions, ["MAKE", "TYPE_VEHICLE", "INSR_TYPE"], tmp_path / "rules.json",
        min_support=0.01, min_confidence=0.5, min_lift=1.5, max_len=2, n_rules=5,
        exclude_majority_consequent=True,
    )
    consequents = {
        item["value"] for rule in rules for item in rule["consequents"]
    }
    # 1202 — мажоритный класс, его быть не должно
    assert "1202" not in consequents


def test_majority_filter_can_be_disabled(tmp_path, transactions):
    """Отключение фильтра допускает тривиальные правила обратно.

    Замечание: ранжирование по lift само по себе уже опускает правила
    в сторону мажоритного класса на последние места, поэтому в небольшой
    выдаче их не видно и без явного фильтра. Чтобы проверить именно фильтр,
    берётся широкая выдача с минимальным lift.
    """
    rules = generate_association_rules(
        transactions, ["MAKE", "TYPE_VEHICLE", "INSR_TYPE"], tmp_path / "rules.json",
        min_support=0.01, min_confidence=0.5, min_lift=1.0, max_len=2, n_rules=20,
        exclude_majority_consequent=False,
    )
    consequents = {item["value"] for rule in rules for item in rule["consequents"]}
    assert "1202" in consequents


def test_rules_ranked_by_lift(tmp_path, transactions):
    """Отбор идёт по lift, и правила отсортированы по убыванию."""
    rules = generate_association_rules(
        transactions, ["MAKE", "TYPE_VEHICLE", "INSR_TYPE"], tmp_path / "rules.json",
        min_support=0.01, min_confidence=0.5, min_lift=1.2, max_len=2, n_rules=5,
    )
    lifts = [rule["lift"] for rule in rules]
    assert lifts == sorted(lifts, reverse=True)
    assert all(rule["lift"] >= 1.2 for rule in rules)


def test_informative_rule_is_found(tmp_path, transactions):
    """Содержательное правило «ISUZU -> грузовик» должно быть найдено."""
    rules = generate_association_rules(
        transactions, ["MAKE", "TYPE_VEHICLE", "INSR_TYPE"], tmp_path / "rules.json",
        min_support=0.01, min_confidence=0.5, min_lift=1.5, max_len=2, n_rules=5,
    )
    pairs = {
        (item["column"], item["value"])
        for rule in rules
        for item in rule["antecedents"] + rule["consequents"]
    }
    assert ("MAKE", "ISUZU") in pairs
    assert ("TYPE_VEHICLE", "Truck") in pairs


def test_rules_roundtrip_through_file(tmp_path, transactions):
    """Записанные правила читаются обратно без потерь."""
    path = tmp_path / "rules.json"
    written = generate_association_rules(
        transactions, ["MAKE", "TYPE_VEHICLE"], path,
        min_support=0.01, min_lift=1.5, n_rules=3,
    )
    loaded = load_association_rules(path)
    assert loaded == written
    assert len(loaded) <= 3


def test_missing_rules_file_returns_empty(tmp_path):
    """Отсутствие файла правил — пустой список, а не исключение."""
    assert load_association_rules(tmp_path / "absent.json") == []


def test_unsupported_algorithm_is_rejected(tmp_path, transactions):
    """Неизвестный алгоритм отвергается с понятной ошибкой."""
    with pytest.raises(ValueError) as error:
        generate_association_rules(
            transactions, ["MAKE"], tmp_path / "r.json", algorithm="punch"
        )
    assert "fpgrowth" in str(error.value)


def test_impossible_thresholds_yield_no_rules(tmp_path, transactions):
    """Заведомо недостижимые пороги дают пустой список, а не сбой."""
    rules = generate_association_rules(
        transactions, ["MAKE", "TYPE_VEHICLE"], tmp_path / "r.json",
        min_support=0.99, min_lift=1000.0, n_rules=5,
    )
    assert rules == []
    assert load_association_rules(tmp_path / "r.json") == []


def test_rule_statistics(tmp_path):
    """Сводка по правилам считается и переживает пустой список."""
    assert rule_statistics([]) == {"n_rules": 0}
    rules = [
        {"label": "a", "antecedents": [], "consequents": [],
         "support": 0.02, "confidence": 0.8, "lift": 4.0},
    ]
    stats = rule_statistics(rules)
    assert stats["n_rules"] == 1
    assert stats["max_lift"] == 4.0


# ---------------------------------------------------------------------------
# Служебные функции
# ---------------------------------------------------------------------------


def test_numpy_scalars_become_json_types():
    """numpy-типы приводятся к типам, допустимым в JSON."""
    payload = {
        "flag": np.bool_(True),
        "count": np.int64(5),
        "value": np.float64(0.25),
        "array": np.array([1, 2, 3]),
        "nan": float("nan"),
        "infinite": float("inf"),
    }
    converted = to_json_serializable(payload)

    assert converted["flag"] is True
    assert converted["count"] == 5 and isinstance(converted["count"], int)
    assert converted["array"] == [1, 2, 3]
    # JSON не допускает NaN и бесконечностей — они становятся null
    assert converted["nan"] is None
    assert converted["infinite"] is None

    # Должно сериализоваться без ошибок
    json.dumps(converted)


def test_timestamps_become_isoformat():
    """Временные метки приводятся к строке, а не падают при dumps."""
    converted = to_json_serializable(
        {"when": pd.Timestamp("2014-07-01"), "days": pd.Timedelta(days=3)}
    )
    assert converted["when"].startswith("2014-07-01")
    assert converted["days"] == 259200.0
    json.dumps(converted)


def test_atomic_pickle_dump_leaves_single_file(tmp_path):
    """После атомарной записи остаётся ровно один файл."""
    path = tmp_path / "model.pkl"
    atomic_pickle_dump({"a": 1, "b": [1, 2, 3]}, path)

    assert [p.name for p in tmp_path.iterdir()] == ["model.pkl"]
    assert safe_pickle_load(path) == {"a": 1, "b": [1, 2, 3]}


def test_safe_pickle_load_explains_corruption(tmp_path):
    """Повреждённый артефакт даёт понятное сообщение, а не EOFError."""
    path = tmp_path / "model.pkl"
    path.write_bytes(b"")  # обрезанный файл

    with pytest.raises(ValueError) as error:
        safe_pickle_load(path)
    assert "повреждён" in str(error.value)
    assert "model.pkl" in str(error.value)


def test_safe_pickle_load_missing_file(tmp_path):
    """Отсутствующий файл — FileNotFoundError, а не ValueError."""
    with pytest.raises(FileNotFoundError):
        safe_pickle_load(tmp_path / "absent.pkl")


def test_load_json_tolerates_garbage(tmp_path):
    """Повреждённый JSON даёт значение по умолчанию, а не исключение."""
    path = tmp_path / "x.json"
    path.write_text("{не json", encoding="utf-8")
    assert load_json(path, default={"fallback": True}) == {"fallback": True}
    assert load_json(tmp_path / "absent.json", default=[]) == []


def test_saved_pickle_matches_manual_dumps(tmp_path):
    """Атомарная запись даёт тот же результат, что и обычный pickle."""
    payload = {"model": [1.5, 2.5], "name": "тест"}
    path = tmp_path / "m.pkl"
    atomic_pickle_dump(payload, path)

    with open(path, "rb") as handle:
        assert pickle.load(handle) == payload
