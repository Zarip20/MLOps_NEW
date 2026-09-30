"""Предобработка данных.

Собирает `ColumnTransformer` по конфигурации, а не по захардкоженным
значениям. Препроцессор обучается на первом батче и далее только
применяется — это и обеспечивает постоянную ширину матрицы, от которой
зависит работа `partial_fit`.

Добавлен `feature_fingerprint`: цифровой отпечаток схемы признаков, который
пишется в манифест и реестр моделей (АР-9). Он позволяет отказаться
работать с моделью, обученной на другой схеме, вместо того чтобы падать
внутри `predict` с невнятным `ValueError`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

NUMERIC_IMPUTERS = ("median", "mean", "most_frequent", "constant")
CATEGORICAL_IMPUTERS = ("most_frequent", "constant")
SCALERS = ("standard", "robust", "none")


def _numeric_imputer(strategy: str) -> SimpleImputer:
    if strategy == "constant":
        return SimpleImputer(strategy="constant", fill_value=0)
    return SimpleImputer(strategy=strategy)


def _scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    return "passthrough"


def create_preprocessor(config: dict[str, Any]) -> ColumnTransformer:
    """Собрать препроцессор по секции `preprocessing` конфигурации.

    Числовые признаки: заполнение пропусков и масштабирование.
    Категориальные: заполнение пропусков и one-hot кодирование с
    `handle_unknown='ignore'`, чтобы новые категории поздних батчей не
    ломали матрицу.
    """
    numerical_cols = list(config["numerical_cols"])
    categorical_cols = list(config["categorical_cols"])

    numeric_imputer = config.get("numeric_imputer", "median")
    if numeric_imputer not in NUMERIC_IMPUTERS:
        raise ValueError(
            f"preprocessing.numeric_imputer должен быть одним из {NUMERIC_IMPUTERS}"
        )

    categorical_imputer = config.get("categorical_imputer", "most_frequent")
    if categorical_imputer not in CATEGORICAL_IMPUTERS:
        raise ValueError(
            f"preprocessing.categorical_imputer должен быть одним из {CATEGORICAL_IMPUTERS}"
        )

    scaler_name = config.get("scaler", "standard")
    if scaler_name not in SCALERS:
        raise ValueError(f"preprocessing.scaler должен быть одним из {SCALERS}")

    top_k = int(config.get("make_top_k", 0) or 0)

    num_steps: list[tuple[str, Any]] = [("imputer", _numeric_imputer(numeric_imputer))]
    if scaler_name != "none":
        num_steps.append(("scaler", _scaler(scaler_name)))

    transformers: list[tuple[str, Any, list[str]]] = [
        ("num", Pipeline(num_steps), numerical_cols),
    ]

    for column in categorical_cols:
        if top_k > 0 and column in config.get("top_k_columns", []):
            encoder = OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=top_k,
                max_categories=top_k,
                sparse_output=False,
                dtype="float32",
            )
        else:
            encoder = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, dtype="float32"
            )
        transformers.append(
            (
                column,
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy=categorical_imputer)),
                        ("onehot", encoder),
                    ]
                ),
                [column],
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_preprocessor(
    preprocessor: ColumnTransformer, frame: pd.DataFrame
) -> pd.DataFrame:
    """Обучить препроцессор и вернуть матрицу признаков."""
    return preprocessor.fit_transform(frame)


def transform(
    preprocessor: ColumnTransformer, frame: pd.DataFrame
) -> pd.DataFrame:
    """Применить обученный препроцессор."""
    return preprocessor.transform(frame)


def input_feature_names(preprocessor: ColumnTransformer) -> dict[str, list[str]]:
    """Какие входные колонки требует обученный препроцессор.

    Препроцессор объявляет собственную схему, и это надёжнее, чем
    пересобирать список из конфигурации: набор производных признаков
    фиксируется на первом батче, а конфигурация может быть изменена
    позже. Без сверки с препроцессором `inference` и **возобновление
    прогона в новом процессе** получали бы матрицу не той ширины —
    сначала это выглядело бы как «ошибка в данных», хотя на деле
    расходилась бы схема.
    """
    result: dict[str, list[str]] = {"num": [], "cat": []}
    # У обученного `ColumnTransformer` записи имеют разную длину:
    # `transformers` — тройки, `transformers_` — четвёрки с весом.
    # Поэтому схема читается по позициям, а не распаковкой кортежа.
    try:
        transformers = getattr(preprocessor, "transformers_", None)
        if transformers is None:
            transformers = preprocessor.transformers
    except (AttributeError, TypeError):
        return result

    for entry in transformers:
        if entry is None:
            continue
        if isinstance(entry, (list, tuple)) and len(entry) < 3:
            continue
        name = entry[0]
        columns = entry[2] if len(entry) > 2 else None
        if columns is None or isinstance(columns, str):
            continue
        columns = list(columns)
        if name == "num":
            result["num"].extend(columns)
        else:
            result["cat"].extend(columns)
    return result


def known_categories(preprocessor: ColumnTransformer) -> dict[str, list[str]]:
    """Значения категорий, которые препроцессор видел при обучении.

    Берутся из обученного `OneHotEncoder`, а не из метаданных качества:
    там хранится лишь несколько самых частых значений, и для `MAKE` с
    сотнями марок этого хватило бы на то, чтобы пометить нормальные
    значения как незнакомые. Кодировщик же знает ровно тот набор, на
    котором обучался, — тот самый, относительно которого «незнакомое»
    и имеет смысл.

    Returns:
        Словарь «колонка → известные значения»; без обученных
        кодировщиков пустой.
    """
    result: dict[str, list[str]] = {}
    try:
        entries = getattr(preprocessor, "transformers_", None)
        if entries is None:
            entries = preprocessor.transformers
    except (AttributeError, TypeError):
        return result

    for entry in entries:
        if entry is None or not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        columns = entry[2]
        steps = entry[1]
        if columns is None or isinstance(columns, str):
            continue
        columns = list(columns)
        if isinstance(steps, Pipeline):
            steps = list(steps.named_steps.values())
        elif not isinstance(steps, (list, tuple)):
            steps = [steps]
        for step in steps:
            categories = getattr(step, "categories_", None)
            if not categories:
                continue
            for name, values in zip(columns, categories):
                result[str(name)] = [str(item) for item in values]
    return result


def verify_feature_schema(
    expected: dict[str, list[str]], frame: pd.DataFrame
) -> list[str]:
    """Столбцы, которых не хватает во входном датафрейме."""
    required = [*expected.get("num", []), *expected.get("cat", [])]
    return [column for column in required if column not in frame.columns]


def feature_fingerprint(preprocessor: ColumnTransformer) -> dict[str, Any]:
    """Отпечаток схемы признаков обученного препроцессора (АР-9).

    Включает число выходных признаков, состав входных колонок и версию
    scikit-learn: при несовпадении модель и препроцессор использовать нельзя.
    """
    output_features = list(getattr(preprocessor, "get_feature_names_out", lambda: [])())

    try:
        transformers = dict(preprocessor.transformers)
    except (TypeError, ValueError):
        transformers = {}

    return {
        "n_features": len(output_features),
        "feature_names": output_features,
        "columns": {
            name: list(spec[2]) if len(spec) > 2 and spec[2] is not None else []
            for name, spec in transformers.items()
        },
        "sklearn_version": sklearn.__version__,
    }


def fingerprint_hash(fingerprint: dict[str, Any]) -> str:
    """Короткий хэш отпечатка для сравнения в реестре."""
    payload = json.dumps(
        {
            "n_features": fingerprint.get("n_features"),
            "columns": fingerprint.get("columns"),
            "sklearn_version": fingerprint.get("sklearn_version"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
