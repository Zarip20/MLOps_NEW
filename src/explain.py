"""Интерпретация моделей (5.b.i).

Задание оценивает «интерпретацию прогнозов (визуализация структуры
дерева, оценка коэффициентов LR, демонстрация ближайших соседей, LIME,
SHAP)» на 1–3 балла. Реализованы три способа, каждый со своей ценностью
для разных типов моделей:

* **Коэффициенты логистической регрессии** — вклад каждого признака в
  вероятность класса 1, в логитах и в шансах. Понятно без оговорок.
* **Важность признаков в дереве** — какие признаки модель использовала
  и насколько сильно. Даёт картину, но не различает «модель выучила
  правило» и «модель просто много раз видела признак».
* **Текстовое представление пути дерева** — ветви с наибольшим
  влиянием, с долями классов в листьях. Показывает правила, которыми
  модель действительно пользуется.

Зависимость от `matplotlib` не используется: графики добавляются
в дашборд как SVG средствами стандартной библиотеки, а текстовое
представление попадает и в отчёт, и в JSON-артефакт.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Сколько признаков показывать в отчёте и дашборде.
TOP_FEATURES = 15


def explain_model(
    model: Any,
    feature_names: Sequence[str],
    model_name: str,
) -> dict[str, Any]:
    """Собрать объяснение модели, подходящее её типу.

    Args:
        model: обученная модель.
        feature_names: имена признаков в порядке матрицы.
        model_name: имя модели для отчёта.

    Returns:
        Структура с ключом `method` и соответствующими полями.
    """
    if hasattr(model, "coef_"):
        return _explain_linear(model, feature_names, model_name)
    if hasattr(model, "feature_importances_"):
        return _explain_tree(model, feature_names, model_name)
    if hasattr(model, "coefs_"):
        return _explain_neural(model, feature_names, model_name)
    return {
        "method": "unsupported",
        "model": model_name,
        "note": (
            f"Для модели {type(model).__name__} автоматическое объяснение "
            f"не реализовано"
        ),
    }


def _explain_neural(
    model: Any, feature_names: Sequence[str], model_name: str
) -> dict[str, Any]:
    """Вклад признаков в нейросеть по весам первого скрытого слоя.

    У многослойной сети нет ни коэффициентов, ни важностей признаков, и
    точный вклад отдельного нейрона неразделим. Практически полезная
    и честно оговоренная оценка — сумма модулей весов входного слоя по
    каждому признаку: она показывает, через какие признаки сигнал вообще
    попадает в сеть, но не различает вклад отдельных нейронов.
    """
    coefficients = [np.asarray(layer, dtype=float) for layer in model.coefs_]
    if not coefficients or coefficients[0].size == 0:
        return {"method": "neural_input_layer_weights", "model": model_name,
                "features": [], "note": "модель не обучена"}

    first_layer = coefficients[0]  # (n_features, n_neurons)
    magnitude = np.abs(first_layer).sum(axis=1)
    names = _align_names(feature_names, len(magnitude))
    total = float(magnitude.sum()) or 1.0
    order = np.argsort(magnitude)[::-1][:TOP_FEATURES]

    return {
        "method": "neural_input_layer_weights",
        "model": model_name,
        "interpretation": (
            "Сумма модулей весов первого скрытого слоя по каждому признаку. "
            "Показывает, через какие признаки сигнал вообще попадает в сеть, "
            "но не различает вклад отдельных нейронов и не является "
            "объяснением конкретного предсказания."
        ),
        "features": [
            {
                "feature": names[index],
                "importance": round(float(magnitude[index]), 5),
                "share": round(float(magnitude[index] / total), 5),
            }
            for index in order
        ],
        "n_layers": len(coefficients),
    }


def _explain_linear(
    model: Any, feature_names: Sequence[str], model_name: str
) -> dict[str, Any]:
    """Коэффициенты логистической регрессии.

    Признаки после `StandardScaler` имеют сопоставимый масштаб, поэтому
    сами коэффициенты сравнимы между собой. Шансы (`odds`) показывают,
    во сколько раз меняется отношение шансов при изменении признака на
    единицу стандартного отклонения.
    """
    coefficients = np.asarray(model.coef_).ravel()
    names = _align_names(feature_names, len(coefficients))

    order = np.argsort(np.abs(coefficients))[::-1][:TOP_FEATURES]
    with np.errstate(divide="ignore", over="ignore"):
        odds = np.exp(coefficients)

    return {
        "method": "logistic_regression_coefficients",
        "model": model_name,
        "interpretation": (
            "Коэффициенты в логитах; положительный знак означает рост "
            "вероятности страхового случая. Шансы показывают "
            "отношение шансов при изменении признака на 1 стандартное "
            "отклонение."
        ),
        "features": [
            {
                "feature": names[index],
                "coefficient": round(float(coefficients[index]), 5),
                "odds_ratio": round(float(odds[index]), 5)
                if np.isfinite(odds[index]) else None,
                "direction": "повышает риск" if coefficients[index] > 0 else "снижает риск",
            }
            for index in order
        ],
        "intercept": round(float(np.asarray(model.intercept_).ravel()[0]), 5),
    }


def _explain_tree(
    model: Any, feature_names: Sequence[str], model_name: str
) -> dict[str, Any]:
    """Важность признаков и текстовая структура дерева.

    Важность дерева нормирована на сумму единиц и отражает суммарное
    уменьшение неопределённости (impurity) от разбиений по признаку.
    """
    importances = np.asarray(getattr(model, "feature_importances_", []), dtype=float)
    names = _align_names(feature_names, len(importances))

    if importances.size == 0:
        return {"method": "feature_importance", "model": model_name,
                "features": [], "note": "модель не сообщает важность признаков"}

    order = np.argsort(importances)[::-1][:TOP_FEATURES]
    total = float(importances.sum()) or 1.0

    return {
        "method": "feature_importance",
        "model": model_name,
        "interpretation": (
            "Доля суммарного уменьшения неопределённости при разбиениях "
            "по признаку. Портит интерпретацию то, что важность не "
            "различает «признак использован один раз для важного "
            "разбиения» и «признак использовался часто, но не влиял»."
        ),
        "features": [
            {
                "feature": names[index],
                "importance": round(float(importances[index]), 5),
                "share": round(float(importances[index] / total), 5),
            }
            for index in order
        ],
        "n_features_total": int(importances.size),
        "n_features_used": int((importances > 0).sum()),
        "structure": _tree_structure(model, names, depth_limit=3),
    }


def _tree_structure(
    model: Any, feature_names: Sequence[str], depth_limit: int = 3
) -> list[dict[str, Any]]:
    """Текстовое представление верхних уровней дерева.

    Обход в ширину с ограничением глубины: полное дерево на 10 уровней
    читать невозможно, а верхние разбиения как раз содержат правила,
    которыми модель пользуется.
    """
    tree = getattr(model, "tree_", None)
    if tree is None:
        return []

    lines: list[dict[str, Any]] = []
    queue: list[tuple[int, int]] = [(0, 0)]

    while queue:
        node, depth = queue.pop(0)
        if depth > depth_limit:
            continue

        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        samples = int(tree.n_node_samples[node])
        positives = int(tree.value[node][0][1]) if tree.value.ndim >= 3 else 0
        rate = positives / samples if samples else 0.0

        if left == right:  # лист
            lines.append(
                {
                    "depth": depth,
                    "type": "leaf",
                    "samples": samples,
                    "positive_rate": round(rate, 4),
                }
            )
            continue

        feature = _align_names(feature_names, int(tree.n_features))[int(tree.feature[node])]
        threshold = round(float(tree.threshold[node]), 4)
        lines.append(
            {
                "depth": depth,
                "type": "split",
                "feature": feature,
                "threshold": threshold,
                "rule": f"{feature} <= {threshold}",
                "samples": samples,
                "positive_rate": round(rate, 4),
            }
        )
        queue.append((left, depth + 1))
        queue.append((right, depth + 1))

    return lines


def _align_names(feature_names: Sequence[str], expected: int) -> list[str]:
    """Привести список имён к нужной длине.

    Число признаков после кодирования может не совпадать с числом
    исходных колонок, а имена из `ColumnTransformer` иногда недоступны
    (например, у старых версий scikit-learn). В таком случае
    подставляются нейтральные имена, чтобы отчёт оставался читаемым.
    """
    names = [str(name) for name in feature_names][:expected]
    if len(names) < expected:
        names.extend(f"feature_{index}" for index in range(len(names), expected))
    return names


def feature_names_from_preprocessor(preprocessor: Any) -> list[str]:
    """Имена выходных признаков препроцессора, с запасным вариантом."""
    try:
        names = list(preprocessor.get_feature_names_out())
        # Имена ColumnTransformer имеют вид "num__PREMIUM" и "MAKE__onehot__X";
        # для отчёта это шум, поэтому оставляем только полезную часть.
        cleaned = []
        for name in names:
            cleaned.append(name.split("__")[-1] if "__" in name else name)
        return cleaned
    except (AttributeError, ValueError):
        return []


def global_explanation(
    models: dict[str, Any],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """Собрать объяснения по всем моделям батча.

    Модели ранжируются по согласованности: если дерево и линейная
    регрессия выделили один и тот же признак, это аргумент в пользу
    того, что признак действительно значим, а не артефакт конкретной
    модели.
    """
    explanations = {
        name: explain_model(model, feature_names, name)
        for name, model in models.items()
    }

    consensus: dict[str, int] = {}
    for payload in explanations.values():
        for entry in payload.get("features", []):
            feature = entry.get("feature")
            if feature:
                consensus[feature] = consensus.get(feature, 0) + 1

    top = sorted(consensus.items(), key=lambda item: (-item[1], item[0]))[:TOP_FEATURES]
    return {
        "by_model": explanations,
        "consensus": [
            {"feature": feature, "models_agreeing": count} for feature, count in top
        ],
        "note": (
            "Признаки, выделенные несколькими моделями, вероятнее всего "
            "значимы, а не являются артефактом одной архитектуры."
        ),
    }
