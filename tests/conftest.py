"""Общие приспособления для тестов.

Тесты не должны зависеть от реального набора данных: используются небольшие
синтетические датафреймы, собранные так, чтобы воспроизвести характерные
особенности исходных данных — пропуски, сентинельные нули, разный регистр
одних и тех же категорий, грузовые типы кузова.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rules import RuleSet  # noqa: E402

NUMERICAL = ["INSURED_VALUE", "PREMIUM", "PROD_YEAR", "SEATS_NUM", "CARRYING_CAPACITY"]
CATEGORICAL = ["SEX", "INSR_TYPE", "TYPE_VEHICLE", "MAKE", "USAGE"]
TARGET = "CLAIM_PAID"
TARGET_NAME = "HAS_CLAIM"


@pytest.fixture
def sample_batch() -> pd.DataFrame:
    """Батч на 20 строк со всеми особенностями, важными для проверок.

    Особенности:
    * `Truck` с пустой грузоподъёмностью — должно нарушать правило;
    * `Truck` с нулевой грузоподъёмностью — тоже;
    * `Truck` с корректной — не должно;
    * `Motor-cycle` с пустой грузоподъёмностью — не должно (правило условное);
    * `Special construction` в TYPE_VEHICLE и `Special Construction` в USAGE —
      расхождение регистра, на котором проверяется нормализация метрики;
    * пропуски в числовых колонках — чтобы отличать пропуск от нарушения;
    * нулевые `INSURED_VALUE` — сентинельные значения, они не нарушение.
    """
    rows: list[dict] = []

    # 6 грузовиков с корректной грузоподъёмностью
    for index in range(6):
        rows.append(
            {
                "SEX": "0", "INSR_TYPE": "1202", "INSURED_VALUE": 100000.0 + index,
                "PREMIUM": 500.0 + index, "PROD_YEAR": 2015, "SEATS_NUM": 2,
                "CARRYING_CAPACITY": 1500.0, "TYPE_VEHICLE": "Truck",
                "MAKE": "ISUZU", "USAGE": "Own Goods", "INSR_BEGIN": "08-AUG-15",
                "CLAIM_PAID": "01-SEP-15",
            }
        )
    # 4 грузовика с пустой грузоподъёмностью -> нарушение
    for index in range(4):
        rows.append(
            {
                "SEX": "0", "INSR_TYPE": "1202", "INSURED_VALUE": 90000.0 + index,
                "PREMIUM": 400.0 + index, "PROD_YEAR": 2014, "SEATS_NUM": 2,
                "CARRYING_CAPACITY": None, "TYPE_VEHICLE": "Truck",
                "MAKE": "ISUZU", "USAGE": "Own Goods", "INSR_BEGIN": "08-AUG-15",
                "CLAIM_PAID": None,
            }
        )
    # 3 грузовика с нулевой грузоподъёмностью -> нарушение
    for index in range(3):
        rows.append(
            {
                "SEX": "0", "INSR_TYPE": "1202", "INSURED_VALUE": 80000.0 + index,
                "PREMIUM": 300.0 + index, "PROD_YEAR": 2016, "SEATS_NUM": 3,
                "CARRYING_CAPACITY": 0.0, "TYPE_VEHICLE": "Truck",
                "MAKE": "DAF", "USAGE": "General Cartage", "INSR_BEGIN": "08-AUG-15",
                "CLAIM_PAID": None,
            }
        )
    # 4 мотоцикла с пустой грузоподъёмностью -> НЕ нарушение.
    # Нулевой INSURED_VALUE здесь намеренно повторяется: это проверка того,
    # что сентинельный ноль не считается нарушением. Уникальность строк
    # обеспечивается другими колонками, иначе дедупликация в очистке
    # скрыла бы поведение правил.
    for index in range(4):
        rows.append(
            {
                "SEX": "1", "INSR_TYPE": "1201", "INSURED_VALUE": 0.0,
                "PREMIUM": 100.0 + index, "PROD_YEAR": 2017, "SEATS_NUM": 1,
                "CARRYING_CAPACITY": None, "TYPE_VEHICLE": "Motor-cycle",
                "MAKE": "BAJAJ", "USAGE": "Private", "INSR_BEGIN": "08-AUG-15",
                "CLAIM_PAID": None,
            }
        )
    # 3 записи с расхождением регистра категорий
    for index in range(3):
        rows.append(
            {
                "SEX": "0", "INSR_TYPE": "1202", "INSURED_VALUE": 45000.0 + index,
                "PREMIUM": 250.0 + index, "PROD_YEAR": 2013, "SEATS_NUM": None,
                "CARRYING_CAPACITY": 10.0, "TYPE_VEHICLE": "Special construction",
                "MAKE": "AFRO", "USAGE": "Special Construction", "INSR_BEGIN": "08-AUG-15",
                "CLAIM_PAID": "05-OCT-15",
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture
def rule_set() -> RuleSet:
    """Набор правил, повторяющий рабочий config.yaml."""
    return RuleSet.from_config(
        [
            {
                "name": "premium_non_negative",
                "column": "PREMIUM",
                "op": "ge",
                "value": 0,
                "on_missing": "ignore",
                "severity": "error",
            },
            {
                "name": "insured_value_non_negative",
                "column": "INSURED_VALUE",
                "op": "ge",
                "value": 0,
                "on_missing": "ignore",
                "severity": "error",
                "description": "Ноль — значимое значение, а не нарушение",
            },
            {
                "name": "seats_non_negative",
                "column": "SEATS_NUM",
                "op": "ge",
                "value": 0,
                "on_missing": "ignore",
                "severity": "error",
            },
            {
                "name": "cargo_capacity_required",
                "column": "CARRYING_CAPACITY",
                "op": "gt",
                "value": 0,
                "on_missing": "violation",
                "severity": "error",
                "applies_when": {
                    "column": "TYPE_VEHICLE",
                    "op": "in",
                    "value": ["Truck", "Tanker", "Trailers and semitrailers"],
                },
            },
            {
                "name": "make_watch",
                "column": "MAKE",
                "op": "notna",
                "on_missing": "ignore",
                "severity": "warn",
            },
        ]
    )


@pytest.fixture
def association_rules() -> list[dict]:
    """Правила в том формате, который сохраняет `src/association.py`."""
    return [
        {
            "label": "TYPE_VEHICLE = Special construction -> USAGE = Special Construction",
            "antecedents": [{"column": "TYPE_VEHICLE", "value": "Special construction"}],
            "consequents": [{"column": "USAGE", "value": "Special Construction"}],
            "support": 0.02,
            "confidence": 0.6,
            "lift": 24.9,
        },
        {
            "label": "MAKE = ISUZU -> TYPE_VEHICLE = Truck",
            "antecedents": [{"column": "MAKE", "value": "ISUZU"}],
            "consequents": [{"column": "TYPE_VEHICLE", "value": "Truck"}],
            "support": 0.01,
            "confidence": 0.99,
            "lift": 22.0,
        },
    ]


def with_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Добавить бинарную метку HAS_CLAIM, как это делает конвейер."""
    result = frame.copy()
    result[TARGET_NAME] = result[TARGET].notna().astype(int)
    return result
