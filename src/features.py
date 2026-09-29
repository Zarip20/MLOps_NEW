"""Автоматический разведочный анализ данных (2.b.i) и признаки (2.b.ii).

Модуль объединяет две связанные задачи:

* **EDA** — что в данных есть, насколько они полны, какие значения
  встречаются редко, есть ли сентинельные нули и разнобой регистра в
  категориях. Результат — JSON на каждый батч, из которого потом
  строится дашборд.
* **Feature Engineering** — производные признаки, которые в исходном
  наборе не выделены, но содержат для модели полезный сигнал:
  возраст автомобиля на момент страхования, длительность полиса, год и
  месяц начала, отношение стоимости к премии, индикаторы сентинельных
  нулей.

Признаки добавляются **до** предобработки, поэтому они проходят через
тот же `ColumnTransformer`, что и исходные: это гарантирует, что
`SimpleImputer` и кодирование категорий к ним применимы, и не требует
отдельной ветки в коде обучения.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.utils import parse_datetime

logger = logging.getLogger(__name__)

#: Порог, начиная с которого значение считается редкой категорией.
RARE_CATEGORY_RATIO = 0.01


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------


def profile_frame(
    frame: pd.DataFrame,
    numerical_cols: Sequence[str],
    categorical_cols: Sequence[str],
    target: str | None = None,
    max_examples: int = 5,
) -> dict[str, Any]:
    """Собрать профиль батча: полнота, распределения, кардинальности.

    Args:
        frame: батч.
        numerical_cols, categorical_cols: состав признаков.
        target: имя целевой колонки, если она уже сформирована.
        max_examples: сколько примеров значений приводить.

    Returns:
        Структура, пригодная для записи в JSON без преобразований.
    """
    profile: dict[str, Any] = {
        "rows": int(len(frame)),
        "n_columns": int(len(frame.columns)),
        "numerical": {},
        "categorical": {},
    }

    for column in numerical_cols:
        if column not in frame.columns:
            continue
        profile["numerical"][column] = _profile_numeric(frame[column], max_examples)

    for column in categorical_cols:
        if column not in frame.columns:
            continue
        profile["categorical"][column] = _profile_categorical(frame[column], max_examples)

    if target and target in frame.columns:
        profile["target"] = _profile_target(frame[target])

    profile["case_inconsistency"] = detect_case_inconsistency(frame, categorical_cols)
    return profile


def _profile_numeric(series: pd.Series, max_examples: int) -> dict[str, Any]:
    """Профиль числовой колонки.

    Отдельно считаются нули: в этом наборе они являются сентинельными
    значениями («неизвестно»), а не настоящими измерениями, и это
    принципиально влияет на выбор модели и импутации.
    """
    present = series.dropna()
    total = max(len(series), 1)
    non_numeric = pd.to_numeric(present, errors="coerce")
    non_numeric_count = int(non_numeric.isna().sum())
    numeric = non_numeric.dropna()

    result: dict[str, Any] = {
        "dtype": str(series.dtype),
        "missing": int(series.isna().sum()),
        "missing_ratio": round(float(series.isna().sum() / total), 6),
        "non_numeric": non_numeric_count,
        "zeros": int((numeric == 0).sum()),
        "zeros_ratio": round(float((numeric == 0).sum() / total), 6),
        "unique": int(series.nunique(dropna=True)),
    }

    if non_numeric_count:
        result["non_numeric_examples"] = [
            str(value) for value in present[non_numeric.isna()].unique()[:max_examples]
        ]

    if not numeric.empty:
        quantiles = numeric.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        result.update(
            {
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "mean": round(float(numeric.mean()), 4),
                "std": round(float(numeric.std()), 4),
                "q01": round(float(quantiles.loc[0.01]), 4),
                "q25": round(float(quantiles.loc[0.25]), 4),
                "median": round(float(quantiles.loc[0.5]), 4),
                "q75": round(float(quantiles.loc[0.75]), 4),
                "q99": round(float(quantiles.loc[0.99]), 4),
                # Сильно скошенные распределения плохо переносятся
                # линейными моделями и StandardScaler.
                "skew": round(float(numeric.skew()), 4),
                "heavy_tailed": bool(numeric.skew() > 2),
            }
        )
    return result


def _profile_categorical(series: pd.Series, max_examples: int) -> dict[str, Any]:
    """Профиль категориальной колонки с перечнем редких значений."""
    present = series.dropna()
    total = max(len(series), 1)
    counts = present.value_counts()
    rare_threshold = max(int(total * RARE_CATEGORY_RATIO), 1)
    rare = counts[counts <= rare_threshold]

    result: dict[str, Any] = {
        "dtype": str(series.dtype),
        "missing": int(series.isna().sum()),
        "missing_ratio": round(float(series.isna().sum() / total), 6),
        "unique": int(len(counts)),
        "top": {str(name): int(value) for name, value in counts.head(max_examples).items()},
        "rare_categories": int(len(rare)),
        "rare_rows": int(rare.sum()),
    }
    if len(counts) and counts.iloc[0] > 0:
        result["top_share"] = round(float(counts.iloc[0] / total), 6)
    return result


def _profile_target(series: pd.Series) -> dict[str, Any]:
    """Распределение целевой метки — база для отслеживания дрейфа."""
    counts = series.value_counts()
    total = max(len(series), 1)
    positive = int(counts.get(1, 0))
    return {
        "positive": positive,
        "negative": int(counts.get(0, 0)),
        "positive_rate": round(positive / total, 6),
        "imbalance_ratio": round(
            float(counts.get(0, 0) / positive), 4
        ) if positive else None,
    }


def detect_case_inconsistency(
    frame: pd.DataFrame, categorical_cols: Sequence[str]
) -> dict[str, list[str]]:
    """Найти значения, различающиеся только регистром.

    Практический случай из этого набора: `TYPE_VEHICLE` содержит
    `Special construction`, а `USAGE` — `Special Construction`. Для
    модели это две несвязанные категории, а для человека — одна сущность,
    и такая пара порождает правило с очень высоким процентом нарушений
    (43 % в прогоне), хотя на самом деле данные непротиворечивы.
    """
    result: dict[str, list[str]] = {}
    for column in categorical_cols:
        if column not in frame.columns:
            continue
        values = frame[column].dropna().astype(str)
        if values.empty:
            continue
        # Группируем по нижнему регистру: расхождение есть только там, где
        # в одной группе встретились РАЗНЫЕ написания. Проверка «сколько
        # раз встречается свёрнутое значение» была бы ошибкой — почти
        # каждая категория встречается больше одного раза.
        folded: dict[str, set[str]] = {}
        for value in values.unique():
            folded.setdefault(value.lower(), set()).add(value)

        conflicting = {
            original
            for variants in folded.values()
            if len(variants) > 1
            for original in variants
        }
        if len(conflicting) > 1:
            result[column] = sorted(conflicting)[:10]
    return result


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------


def engineer_features(
    frame: pd.DataFrame,
    numerical_cols: Sequence[str],
    categorical_cols: Sequence[str],
    time_column: str = "INSR_BEGIN",
    date_format: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Добавить производные признаки.

    Все новые признаки числовые: они не требуют отдельного кодирования и
    корректно проходят через `SimpleImputer` и `StandardScaler`.

    Args:
        date_format: формат дат. Задаётся явно, иначе pandas перебирает
            форматы построчно и предупреждает об этом на каждом батче.
    """
    result = frame.copy()
    added: list[str] = []

    def add_numeric(name: str, values: pd.Series) -> None:
        if name in result.columns:
            return
        result[name] = values.astype("float32")
        added.append(name)

    dates = (
        parse_datetime(result[time_column], date_format)
        if time_column in result.columns
        else None
    )

    # 1. Возраст автомобиля на момент начала страхования.
    #    PROD_YEAR — год выпуска, INSR_BEGIN — начало полиса. Разница
    #    в годах прямо соотносится с риском: старые автомобили аварийнее.
    if dates is not None and "PROD_YEAR" in result.columns:
        age = dates.dt.year - pd.to_numeric(result["PROD_YEAR"], errors="coerce")
        add_numeric("VEHICLE_AGE", age.clip(lower=0))

    # 2. Календарные признаки: в данных выражена сезонность
    #    (доля страховых случаев 11 % в июле против 2.6 % в июне 2018),
    #    поэтому месяц и квартал несут сигнал сам по себе.
    if dates is not None:
        add_numeric("POLICY_MONTH", dates.dt.month)
        add_numeric("POLICY_QUARTER", dates.dt.quarter)
        add_numeric("POLICY_YEAR", dates.dt.year)
        add_numeric("POLICY_DAY_OF_YEAR", dates.dt.dayofyear)

    # 3. Длительность полиса в днях, если доступна дата окончания.
    for end_column in ("INSR_END",):
        if dates is not None and end_column in result.columns:
            ends = parse_datetime(result[end_column], date_format)
            duration = (ends - dates).dt.days
            add_numeric("POLICY_DURATION_DAYS", duration.clip(lower=0))

    # 4. Отношение стоимости к премии. Страховая премия и стоимость
    #    имущества связаны нелинейно; их отношение — безразмерная
    #    величина, устойчивая к общему масштабу записей.
    if {"INSURED_VALUE", "PREMIUM"} <= set(result.columns):
        premium = pd.to_numeric(result["PREMIUM"], errors="coerce")
        insured = pd.to_numeric(result["INSURED_VALUE"], errors="coerce")
        add_numeric("VALUE_TO_PREMIUM", (insured / premium.replace(0, np.nan)))

    # 5. Индикаторы сентинельных нулей. Ноль в INSURED_VALUE встречается
    #    у 45.5 % записей и по решению Р-2 остаётся значимым значением,
    #    но отдельный флаг позволяет модели различать «нулевая стоимость»
    #    и «стоимость не указана» — не подменяя одно другим.
    if "INSURED_VALUE" in result.columns:
        add_numeric(
            "IS_INSURED_VALUE_ZERO",
            (pd.to_numeric(result["INSURED_VALUE"], errors="coerce") == 0).astype("int8"),
        )
    if "CARRYING_CAPACITY" in result.columns:
        add_numeric(
            "IS_CAPACITY_MISSING",
            result["CARRYING_CAPACITY"].isna().astype("int8"),
        )

    # 6. Логарифм тяжёлых хвостов. INSURED_VALUE достигает 67.8 млн при
    #    медиане порядка сотен тысяч, PREMIUM — 7.6 млн. Разброс в три
    #    порядка делает StandardScaler неустойчивым, а логарифм приводит
    #    распределение к сопоставимому виду.
    for column in ("INSURED_VALUE", "PREMIUM", "CARRYING_CAPACITY"):
        if column in result.columns:
            values = pd.to_numeric(result[column], errors="coerce")
            add_numeric(f"LOG1P_{column}", np.log1p(values.clip(lower=0)))

    if added:
        logger.info("Добавлено признаков: %d (%s)", len(added), ", ".join(added))
    return result, added


def engineer_feature_list(
    numerical_cols: Sequence[str],
    added: Sequence[str],
) -> list[str]:
    """Итоговый список числовых признаков с учётом производных."""
    return list(numerical_cols) + [name for name in added if name not in numerical_cols]
