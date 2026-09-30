"""Перебор вариантов предобработки с выбором лучшего (3.b.i).

Требование: «несколько вариантов предобработки (median/mean,
StandardScaler/RobustScaler, top-K по частоте MAKE) и перебор с выбором
лучшего». Модуль строит варианты, обучает на каждом и выбирает победителя
по качеству на отложенной части.

**Что здесь честно, а что нет.** Перебор выполняется на одном батче и
одной дешёвой модели — логистической регрессии. Это осознанное
ограничение, а не приближение «для скорости»: дерево, лес и нейросеть
реагируют на масштаб и импутацию иначе, и выбирать предобработку по
одной линейной модели — значит оптимизировать не то. Поэтому победитель
не навязывается всему прогону молча: результат перебора пишется в
отчёт, и по нему видно, на каких данных и какой моделью выбирали.

Замена на полный перебор всех моделей означала бы умножение времени
прогона на число вариантов — при восьми вариантах и четырёх моделях это
восемь полных обучений вместо одного.

Когда перебор запускается, решает конфигурация (`preprocessing.sweep`).
По умолчанию — на первом батче, где препроцессор обучается заново: там
выбор влияет на всё последующее обучение, и пересматривать его на каждом
батче было бы накладно без пользы.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.config import Config
from src.preprocessing import create_preprocessor
from src.training import evaluate_model
from src.utils import safe_pickle_load

logger = logging.getLogger(__name__)

#: Названия вариантов, которые можно задать в конфигурации.
IMPUTERS = ("median", "mean", "most_frequent")
SCALERS = ("standard", "robust", "none")


def variants(config: Config) -> list[dict[str, Any]]:
    """Собрать варианты предобработки из конфигурации.

    Список задаётся явно, а не перебирается декартовым произведением:
    полный перебор дал бы двадцать четыре комбинации, из которых
    заведомо бессмысленны многие (например, `most_frequent` для
    числовой колонки рядом с `RobustScaler` без единой аномалии).
    Перебор «всё подряд» на 48 батчах занял бы больше времени, чем
    всё обучение, и не дал бы дополнительного знания.
    """
    declared = config.get("preprocessing", {}).get("variants")
    if not declared:
        return [{}]

    result: list[dict[str, Any]] = []
    for position, item in enumerate(declared):
        if not isinstance(item, dict):
            raise ValueError(
                f"preprocessing.variants[{position}] должен быть словарём"
            )
        options = dict(item)
        imputer = options.get("numeric_imputer")
        if imputer is not None and imputer not in IMPUTERS:
            raise ValueError(
                f"preprocessing.variants[{position}].numeric_imputer="
                f"{imputer!r}: допустимо {', '.join(IMPUTERS)}"
            )
        scaler = options.get("scaler")
        if scaler is not None and scaler not in SCALERS:
            raise ValueError(
                f"preprocessing.variants[{position}].scaler={scaler!r}: "
                f"допустимо {', '.join(SCALERS)}"
            )
        top_k = options.get("make_top_k")
        if top_k is not None:
            try:
                options["make_top_k"] = int(top_k)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"preprocessing.variants[{position}].make_top_k должен "
                    f"быть целым числом: {error}"
                ) from error
            if options["make_top_k"] < 0:
                raise ValueError(
                    f"preprocessing.variants[{position}].make_top_k "
                    f"не может быть отрицательным"
                )
        if not options.get("name"):
            options["name"] = "_".join(
                f"{key}={value}" for key, value in sorted(options.items())
                if key in ("numeric_imputer", "scaler", "make_top_k")
            ) or f"variant_{position}"
        result.append(options)
    return result


def build(config: Config, base: dict[str, Any], override: dict[str, Any],
          numerical: Sequence[str], categorical: Sequence[str]):
    """Собрать препроцессор по базовой конфигурации с наложенным вариантом."""
    settings = copy.deepcopy(dict(base))
    settings.update(override)
    settings["numerical_cols"] = list(numerical)
    settings["categorical_cols"] = list(categorical)
    return create_preprocessor(settings)


def evaluate(
    config: Config,
    frame: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    target: str,
    numerical: Sequence[str],
    categorical: Sequence[str],
    model_name: str = "lr",
    model_params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Обучить модель на каждом варианте и вернуть результаты.

    Returns:
        Список записей: имя варианта, метрики, время и число признаков.
        Порядок соответствует `variants(config)`.
    """
    from src.training import MODEL_CLASSES

    features = [*numerical, *categorical]
    base = {k: v for k, v in config.data["preprocessing"].items()
            if k not in ("variants", "sweep")}
    results: list[dict[str, Any]] = []

    for index, override in enumerate(variants(config)):
        started = time.perf_counter()
        try:
            preprocessor = build(config, base, override, numerical, categorical)
            x_train = preprocessor.fit_transform(train[features])
            x_val = preprocessor.transform(val[features])

            model = MODEL_CLASSES[model_name](**(model_params or {}))
            model.fit(x_train, train[target].to_numpy())
            metrics = evaluate_model(
                model_name, model, x_val, val[target].to_numpy()
            )
        except Exception as error:  # noqa: BLE001
            # Вариант, который не построился, должен выпасть из
            # перебора с пометкой, а не оборвать его: остальные варианты
            # остаются сравнимыми между собой.
            logger.warning(
                "Вариант предобработки %s не построился: %s",
                override.get("name", index), error,
            )
            results.append({
                "index": index,
                "name": override.get("name", f"variant_{index}"),
                "status": "failed",
                "error": str(error),
                "seconds": round(time.perf_counter() - started, 3),
            })
            continue

        results.append({
            "index": index,
            "name": override.get("name", f"variant_{index}"),
            "status": "ok",
            "options": {
                key: override[key] for key in
                ("numeric_imputer", "scaler", "make_top_k") if key in override
            },
            "n_features": int(x_train.shape[1]),
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "seconds": round(time.perf_counter() - started, 3),
        })

    return results


def pick(results: Sequence[dict[str, Any]], metric: str = "f1") -> dict[str, Any]:
    """Выбрать лучший вариант.

    Returns:
        Запись победителя. Если ни один вариант не построился,
        возвращается пустая запись с `status: none` — вызывающий код
        обязан это учесть, а не молча взять базовую конфигурацию.
    """
    usable = [
        item for item in results
        if item.get("status") == "ok" and item.get(metric) is not None
    ]
    if not usable:
        return {"status": "none", "reason": "ни один вариант не построился"}

    winner = max(usable, key=lambda item: float(item[metric]))
    # Ничья: побеждает первый из равных, и это указано явно, иначе
    # выбор выглядел бы случайным.
    tied = [item for item in usable if float(item[metric]) == float(winner[metric])]
    return {**winner, "n_tied": len(tied)}


def should_run(config: Config, batch_index: int) -> bool:
    """Запускать ли перебор на этом батче.

    По умолчанию — на первом батче и на каждом сотом. Причина в том,
    что перебор обучается заново на каждом вызове, а препроцессор
    переобучается тоже только при первой инициализации: пересматривать
    выбор на каждом батче — платить время без изменения оснований.
    """
    sweep = config.get("preprocessing", {}).get("sweep", {})
    if not sweep.get("enabled", True):
        return False
    first = int(sweep.get("first_batch", 0))
    every = int(sweep.get("every_n_batches", 0))
    if batch_index == first:
        return True
    if every > 0 and batch_index > first and (batch_index - first) % every == 0:
        return True
    return False
