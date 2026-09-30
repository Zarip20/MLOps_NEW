"""Meta Learning: что в прогоне действительно влияло на качество (7.b.iii).

Задание требует «анализа влияния гиперпараметров (накопленные прогоны →
корреляции/ранжирование) и динамики метрик данных и моделей». Здесь
решаются обе части, причём вторая опирается на первую: пока не сказано,
чем именно обучалась та или иная версия, связывать качество с
настройками нельзя.

**Главная трудность — честная.** В прогоне гиперпараметры заданы один
раз и не меняются: 48 батчей обучает одна и та же конфигурация. По
постоянному признаку нельзя оценить его влияние — в сравнении нет
второй группы. Притворяться, что «важность» посчитана, значит выдать
за анализ цифру, у которой нет смысла.

Поэтому анализ честно делит признаки на три случая:

* **варьируется** — значения встречаются в разных прогонах, и по ним
  можно сравнить группы (например, режим обучения `mlp`: первый батч
  — с нуля, остальные — дообучение);
* **не варьируется** — признак постоянен, вывод один: по накопленным
  прогонам его влияние не оценивается, и оценить его можно, лишь
  изменив конфигурацию;
* **выбросы** — `max_iter_reached`, `n_iter_reached`, `rows_used`,
  `store_rows` меняются от батча к батчам сами собой, и по ним
  оценивается влияние *условий обучения*, а не настроек.

Числовые признаки дополнительно оцениваются по ранговой корреляции
Спирмена с f1: она не требует линейности и устойчива к выбросам, чем
лучше для коротких рядов из 48 точек с сезонностью.

Все утверждения снабжены числом наблюдений. Если наблюдений мало,
вывод помечается как недостаточно обоснованный, а не выдаётся за
результат: на одном батче разницы не видно, и утверждать её —
значит выдумывать.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Наблюдений меньше — сравнение групп не выводится.
MIN_GROUP = 3

#: Наблюдений меньше — корреляция не считается надёжной.
MIN_CORRELATION = 8

#: Метрики качества, по которым строится вывод.
QUALITY_METRICS = ("f1", "precision", "recall", "roc_auc")

#: Числовые признаки, по которым оценивается влияние условий обучения.
#: Отбираются те, что меняются от батча к батчам.
CONDITION_KEYS = (
    "rows_used",
    "store_rows",
    "max_iter_reached",
    "n_iter_reached",
)


# ---------------------------------------------------------------------------
# Статистика
# ---------------------------------------------------------------------------


def _rank(values: Sequence[float]) -> list[float]:
    """Ранги с учётом равных значений (средний ранг).

    Свой ранг без учёта совпадений сделал бы корреляцию Спирмена
    смещённой: у `mlp` достигнутое число итераций повторяется почти на
    каждом батче, и такие серии почти не несут информации о порядке.
    """
    array = np.asarray(values, dtype=float)
    order = array.argsort(kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    ranks[order] = np.arange(1, len(array) + 1, dtype=float)

    # Одинаковые значения получают средний ранг своей группы.
    unique, inverse, counts = np.unique(array, return_inverse=True, return_counts=True)
    for index, count in enumerate(counts):
        if count > 1:
            mask = inverse == index
            ranks[mask] = ranks[mask].mean()
    return ranks.tolist()


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Ранговая корреляция Спирмена; `None`, если вычислить нельзя.

    `None`, а не ноль: нулевая корреляция — это результат, а
    невозможность вычислить — отсутствие данных. Смешивать их нельзя,
    иначе в вывод попадёт «влияния нет».
    """
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.all(np.isclose(x, x[0])) or np.all(np.isclose(y, y[0])):
        return None

    rx, ry = np.asarray(_rank(x)), np.asarray(_rank(y))
    if np.all(np.isclose(rx, rx[0])) or np.all(np.isclose(ry, ry[0])):
        return None

    value = float(np.corrcoef(rx, ry)[0, 1])
    if math.isnan(value):
        return None
    return round(value, 4)


def _stats(values: Sequence[float]) -> dict[str, Any]:
    """Описательная статистика по группе значений."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": round(float(array.mean()), 4),
        "median": round(float(np.median(array)), 4),
        "min": round(float(array.min()), 4),
        "max": round(float(array.max()), 4),
        "std": round(float(array.std(ddof=1)) if array.size > 1 else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# Разбор накопленных прогонов
# ---------------------------------------------------------------------------


def collect_runs(manifests: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Развернуть манифесты батчей в список наблюдений «прогон».

    Наблюдение — это пара (модель, батч) со всеми метриками и настройками.
    Такой вид удобен тем, что дальше анализ одинаков для настроек и для
    условий обучения: различаются только ключи.
    """
    runs: list[dict[str, Any]] = []
    for manifest in manifests:
        index = manifest.get("batch_idx")
        metrics = manifest.get("metrics") or {}
        settings = manifest.get("hyperparameters") or {}
        data = manifest.get("data") or {}
        target = data.get("target") or {}
        preprocessor = manifest.get("preprocessor") or {}
        drift = manifest.get("drift") or {}

        for model, values in metrics.items():
            if not isinstance(values, dict):
                continue
            runs.append({
                "batch_idx": index,
                "model": model,
                "metrics": values,
                "settings": dict(settings.get(model) or {}),
                "n_features": preprocessor.get("n_features"),
                "positive_rate": target.get("positive_rate"),
                "rows_clean": data.get("rows_clean"),
                "drift_status": drift.get("status"),
            })
    return runs


def _groups(
    runs: Sequence[dict[str, Any]], key: str
) -> dict[Any, list[float]]:
    """Разложить значения f1 по значениям признака."""
    groups: dict[Any, list[float]] = {}
    for run in runs:
        value = run["settings"].get(key)
        f1 = (run["metrics"] or {}).get("f1")
        if value is None or f1 is None:
            continue
        groups.setdefault(_hashable(value), []).append(float(f1))
    return groups


def _hashable(value: Any) -> Any:
    """Привести значение к ключу словаря.

    В параметрах встречаются списки (например, `hidden_layer_sizes`),
    а они не хешируются. Преобразование в строку делает их
    сопоставимыми между прогонами и не даёт упасть на `TypeError`.
    """
    if isinstance(value, (list, tuple)):
        return json_safe(value)
    return value


def json_safe(value: Any) -> Any:
    """Значение, пригодное для JSON и для сравнения."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


# ---------------------------------------------------------------------------
# Влияние признака
# ---------------------------------------------------------------------------


def feature_influence(
    runs: Sequence[dict[str, Any]], key: str, metric: str = "f1"
) -> dict[str, Any]:
    """Оценить влияние одного признака на метрику **внутри семейства**.

    Сравнение ведётся только между прогонами одной и той же модели.
    Это не педантизм, а единственный корректный способ: у каждой модели
    свои значения параметров, и сравнение «`max_depth=12` у случайного
    леса против `max_depth=10` у дерева» измеряет разницу между лесом и
    деревом, а не влияние глубины. Такая спутанность даёт правдоподобные
    и бессмысленные выводы: она выглядит как «глубже — значит лучше»,
    а на деле это «лес лучше дерева».

    Returns:
        Словарь с признаком, сравнениями по моделям, разбросом и
        вердиктом: `varied` / `constant` / `insufficient` / `confounded`.
    """
    by_model: dict[str, dict[Any, list[float]]] = {}
    for run in runs:
        value = run["settings"].get(key)
        score = (run["metrics"] or {}).get(metric)
        if value is None or score is None:
            continue
        by_model.setdefault(run["model"], {}).setdefault(
            _hashable(value), []
        ).append(float(score))

    all_values: set[Any] = set()
    for values in by_model.values():
        all_values |= set(values)
    n_values = len({str(value) for value in all_values})

    result: dict[str, Any] = {
        "feature": key,
        "metric": metric,
        "n_runs": sum(
            len(scores) for values in by_model.values() for scores in values.values()
        ),
        "n_distinct": n_values,
        "n_models": len(by_model),
        "scope": "внутри семейства моделей",
    }

    if n_values == 0:
        result["verdict"] = "no_data"
        return result

    if n_values == 1:
        result["verdict"] = "constant"
        result["value"] = next(iter(all_values))
        result["note"] = (
            "Признак постоянен во всех прогонах: по накопленным данным "
            "его влияние не оценивается"
        )
        return result

    # Сравнения внутри моделей: у каждой — свои значения.
    comparisons: list[dict[str, Any]] = []
    confounded: list[str] = []
    for model, values in sorted(by_model.items()):
        if len(values) < 2:
            confounded.append(model)
            continue
        summarised = {
            str(name): _stats(scores) for name, scores in values.items()
        }
        small = [n for n, s in summarised.items() if s["n"] < MIN_GROUP]
        means = {n: s["mean"] for n, s in summarised.items() if s["n"]}
        best = max(means, key=lambda n: means[n])
        worst = min(means, key=lambda n: means[n])
        comparisons.append({
            "model": model,
            "groups": summarised,
            "best_value": _unwrap(best),
            "best_mean": means[best],
            "worst_value": _unwrap(worst),
            "worst_mean": means[worst],
            "spread": round(means[best] - means[worst], 4),
            "sufficient": not small,
            "note": (
                f"наблюдений в группе меньше {MIN_GROUP} "
                f"({', '.join(sorted(small))})"
                if small else ""
            ),
        })

    result["comparisons"] = comparisons
    result["not_compared"] = confounded

    if not comparisons:
        result["verdict"] = "confounded"
        result["note"] = (
            "Значения признака совпадают с границами между моделями: "
            f"у {', '.join(sorted(confounded))} оно одно, сравнивать нечего"
        )
        return result

    usable = [item for item in comparisons if item["sufficient"]]
    if not usable:
        result["verdict"] = "insufficient"
        # Называются конкретные группы: по названию видно, чего именно
        # не хватило, и это подсказывает, что пересобрать.
        thin = [
            f"{item['model']}: {item['note']}"
            for item in comparisons if item["note"]
        ]
        result["note"] = (
            "во всех моделях слишком мало наблюдений в группах"
            + (" — " + "; ".join(thin) if thin else "")
        )
        return result

    ranked = sorted(usable, key=lambda item: item["spread"], reverse=True)
    for position, item in enumerate(ranked, start=1):
        # Имя признака проставляется здесь, а не только в `analyse`:
        # функция должна быть самодостаточной и при прямом вызове.
        item["feature"] = key
        item["rank"] = position
    result["verdict"] = "varied"
    result["ranked"] = ranked
    return result


def _unwrap(label: str) -> Any:
    """Значение группы по её строковому имени."""
    if label.startswith("[") and label.endswith("]"):
        try:
            import json as _json

            return _json.loads(label)
        except (ValueError, TypeError):
            return label
    return label


def condition_influence(
    runs: Sequence[dict[str, Any]], metric: str = "f1"
) -> list[dict[str, Any]]:
    """Связь качества с условиями обучения (числовые признаки).

    Это не про настройки, а про то, в каких условиях обучалась модель:
    объём накопленных данных, достигнутое число итераций, доля положительного
    класса. Такие признаки меняются от батча к батчачу сами собой, поэтому
    по ним корреляция считается, в отличие от заданных конфигурацией.
    """
    results: list[dict[str, Any]] = []

    for key in CONDITION_KEYS:
        xs: list[float] = []
        ys: list[float] = []
        for run in runs:
            value = run["settings"].get(key)
            score = (run["metrics"] or {}).get(metric)
            if value is None or score is None:
                continue
            xs.append(float(value))
            ys.append(float(score))
        if len(xs) >= MIN_CORRELATION:
            value = spearman(xs, ys)
            results.append({
                "feature": key,
                "metric": metric,
                "n": len(xs),
                "spearman": value,
                "verdict": "computed" if value is not None else "undefined",
            })

    for key, getter in (
        ("positive_rate", lambda run: run.get("positive_rate")),
        ("n_features", lambda run: run.get("n_features")),
        ("rows_clean", lambda run: run.get("rows_clean")),
        ("batch_idx", lambda run: run.get("batch_idx")),
    ):
        xs: list[float] = []
        ys: list[float] = []
        for run in runs:
            value = getter(run)
            score = (run["metrics"] or {}).get(metric)
            if value is None or score is None:
                continue
            xs.append(float(value))
            ys.append(float(score))
        if len(xs) >= MIN_CORRELATION:
            value = spearman(xs, ys)
            results.append({
                "feature": key,
                "metric": metric,
                "n": len(xs),
                "spearman": value,
                "verdict": "computed" if value is not None else "undefined",
            })

    return [item for item in results if item["verdict"] == "computed"]


# ---------------------------------------------------------------------------
# Динамика
# ---------------------------------------------------------------------------


def dynamics(manifests: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Динамика метрик данных и моделей по батчам."""
    by_model: dict[str, list[dict[str, Any]]] = {}
    data_series: list[dict[str, Any]] = []

    for manifest in manifests:
        index = manifest.get("batch_idx")
        metrics = manifest.get("metrics") or {}
        target = (manifest.get("data") or {}).get("target") or {}
        drift = manifest.get("drift") or {}

        for model, values in metrics.items():
            if not isinstance(values, dict):
                continue
            by_model.setdefault(model, []).append({
                "batch_idx": index,
                "f1": values.get("f1"),
                "roc_auc": values.get("roc_auc"),
                "precision": values.get("precision"),
                "recall": values.get("recall"),
                "predicted_positive_rate": values.get("predicted_positive_rate"),
            })

        data_series.append({
            "batch_idx": index,
            "positive_rate": target.get("positive_rate"),
            "rows_clean": (manifest.get("data") or {}).get("rows_clean"),
            "max_psi": drift.get("max_psi"),
            "drift_status": drift.get("status"),
        })

    models: dict[str, Any] = {}
    for model, series in sorted(by_model.items()):
        series.sort(key=lambda item: (item["batch_idx"] is None, item["batch_idx"]))
        f1 = [item["f1"] for item in series if item["f1"] is not None]
        if not f1:
            continue
        best = max(series, key=lambda item: item["f1"] or 0.0)
        first = series[0]
        last = series[-1]
        models[model] = {
            "n_batches": len(series),
            "f1": _stats(f1),
            "best": {"batch_idx": best["batch_idx"], "f1": best["f1"]},
            "first": {"batch_idx": first["batch_idx"], "f1": first["f1"]},
            "last": {"batch_idx": last["batch_idx"], "f1": last["f1"]},
            "trend": _trend(f1),
        }

    data_series.sort(key=lambda item: (item["batch_idx"] is None, item["batch_idx"]))
    rates = [
        item["positive_rate"] for item in data_series
        if item["positive_rate"] is not None
    ]

    # Победителем по батчу считается модель с максимальным f1. Считается
    # здесь, а не по уже агрегированным моделям: по агрегатам видна
    # лишь средняя картина, а сравнивать надо те значения, которые
    # реально получились на одном и том же батче.
    top: dict[Any, str] = {}
    for manifest in manifests:
        index = manifest.get("batch_idx")
        best_name, best_f1 = None, None
        for model, values in (manifest.get("metrics") or {}).items():
            if not isinstance(values, dict):
                continue
            f1 = values.get("f1")
            if f1 is None:
                continue
            if best_f1 is None or f1 > best_f1:
                best_name, best_f1 = model, f1
        if best_name is not None:
            top[index] = best_name

    wins: dict[str, int] = {}
    for name in top.values():
        wins[name] = wins.get(name, 0) + 1

    return {
        "models": models,
        "data": data_series,
        "positive_rate": _stats(rates) if rates else {"n": 0},
        "best_by_batch": [
            {"batch_idx": index, "model": name}
            for index, name in sorted(
                top.items(), key=lambda pair: (pair[0] is None, pair[0])
            )
        ],
        "wins": {name: count for name, count in sorted(wins.items()) if count},
    }


def _trend(values: Sequence[float]) -> str:
    """Направление динамики по разности первой и последней четверти.

    Линейная регрессия дала бы «наклон», который на сезонном ряду из
    48 точек выглядит убедительно и означает немного: доля положительного
    класса падает, но с сильной сезонностью, и тренд нужно описывать
    словами, а не числом с двумя знаками.
    """
    if len(values) < 8:
        return "мало данных"
    quarter = max(1, len(values) // 4)
    head = float(np.mean(values[:quarter]))
    tail = float(np.mean(values[-quarter:]))
    delta = tail - head
    scale = max(abs(head), 1e-9)
    if abs(delta) / scale < 0.10:
        return "стабильно"
    return "растёт" if delta > 0 else "падает"


# ---------------------------------------------------------------------------
# Итоговый анализ
# ---------------------------------------------------------------------------


def analyse(
    manifests: Sequence[dict[str, Any]],
    metric: str = "f1",
    min_runs: int = MIN_GROUP,
) -> dict[str, Any]:
    """Полный анализ накопленных прогонов.

    Args:
        manifests: манифесты батчей, любой порядок — он приводится к
            хронологическому сам.
        metric: метрика, по которой оценивается влияние.
        min_runs: с какого числа батчей анализ считается осмысленным.

    Returns:
        JSON-совместимый словарь. При малом числе прогонов возвращается
        с явной пометкой, что выводов пока нет: лучше пустой честный
        результат, чем уверенный вывод по двум точкам.
    """
    ordered = sorted(
        manifests,
        key=lambda m: (m.get("batch_idx") is None, m.get("batch_idx")),
    )
    runs = collect_runs(ordered)

    payload: dict[str, Any] = {
        "n_batches": len(ordered),
        "n_runs": len(runs),
        "metric": metric,
        "methods": {
            "group_comparison": (
                "сравнение средних по значениям признака; вывод при "
                f"не менее {min_runs} наблюдениях в каждой группе"
            ),
            "correlation": (
                f"ранговая корреляция Спирмена, не менее "
                f"{MIN_CORRELATION} наблюдений"
            ),
        },
    }

    if len(ordered) < min_runs:
        payload["verdict"] = "insufficient"
        payload["note"] = (
            f"Прогонов {len(ordered)}, нужно не менее {min_runs}: "
            "выводы о влиянии были бы голословными"
        )
        return payload

    # 1. Влияние настроек — только внутри семейств моделей.
    settings_keys: list[str] = []
    for run in runs:
        for key in run["settings"]:
            if key not in settings_keys:
                settings_keys.append(key)

    influence = [
        feature_influence(runs, key, metric)
        for key in sorted(settings_keys)
        if key not in CONDITION_KEYS
    ]
    varied = [item for item in influence if item["verdict"] == "varied"]
    constant = [item for item in influence if item["verdict"] == "constant"]
    weak = [
        item for item in influence
        if item["verdict"] in ("insufficient", "confounded")
    ]

    # Ранжирование — по наибольшему разбросу внутри семейства.
    ranked: list[dict[str, Any]] = []
    for item in varied:
        for entry in item.get("ranked", []):
            ranked.append({**entry, "feature": item["feature"]})
    ranked.sort(key=lambda item: item.get("spread", 0.0), reverse=True)
    for position, item in enumerate(ranked, start=1):
        item["rank"] = position

    payload["settings_recorded"] = bool(settings_keys)
    payload["settings_influence"] = {
        "scope": "сравнение только внутри одной модели",
        "varied": ranked,
        "constant": [
            {"feature": item["feature"], "value": item.get("value")}
            for item in constant
        ],
        "not_evaluated": [
            {
                "feature": item["feature"],
                "verdict": item["verdict"],
                "note": item.get("note"),
            }
            for item in weak
        ],
    }

    # 2. Влияние условий обучения.
    payload["condition_influence"] = sorted(
        condition_influence(runs, metric),
        key=lambda item: abs(item.get("spearman") or 0.0),
        reverse=True,
    )

    # 3. Динамика.
    payload["dynamics"] = dynamics(ordered)
    payload["findings"] = _findings(payload, runs, metric)
    payload["verdict"] = "ok"
    return payload


def _findings(
    payload: dict[str, Any], runs: Sequence[dict[str, Any]], metric: str
) -> list[str]:
    """Выводы словами — то, что читает человек, а не машина."""
    findings: list[str] = []
    dyn = payload.get("dynamics") or {}

    wins = dyn.get("wins") or {}
    if wins:
        best_name, best_count = max(wins.items(), key=lambda pair: pair[1])
        total = sum(wins.values())
        findings.append(
            f"Чаще всего лучшей по {metric} была модель {best_name}: "
            f"{best_count} из {total} батчей"
        )

    for item in (payload.get("settings_influence") or {}).get("varied", []):
        findings.append(
            f"внутри {item['model']}: {item['feature']} = "
            f"{item['best_value']} даёт средний {metric} {item['best_mean']}, "
            f"{item['worst_value']} — {item['worst_mean']} "
            f"(разброс {item['spread']})"
        )

    not_evaluated = (payload.get("settings_influence") or {}).get(
        "not_evaluated"
    ) or []
    confounded = [
        item for item in not_evaluated if item.get("verdict") == "confounded"
    ]
    if confounded:
        findings.append(
            "сравнить нельзя, значения совпадают с границами между моделями: "
            + ", ".join(sorted(item["feature"] for item in confounded))
        )

    constant = (payload.get("settings_influence") or {}).get("constant") or []
    if constant:
        names = ", ".join(sorted(item["feature"] for item in constant))
        findings.append(
            f"настройки, которые не менялись и потому не проверены на "
            f"влияние: {names}"
        )

    conditions = payload.get("condition_influence") or []
    for item in conditions[:3]:
        value = item.get("spearman")
        if value is None:
            continue
        magnitude = abs(value)
        if magnitude >= 0.5:
            strength = "сильная"
        elif magnitude >= 0.25:
            strength = "умеренная"
        else:
            strength = "слабая"
        # Направление называется только когда связь выражена. Фраза
        # «связь −0.01 (падает)» бессмысленна: на таком значении
        # знак не отличим от шума, и читатель примет его за вывод.
        direction = ""
        if magnitude >= 0.25:
            direction = ", растёт" if value > 0 else ", падает"
        findings.append(
            f"{item['feature']}: {strength} связь с {metric}, "
            f"ρ = {value:+.2f}{direction} на {item['n']} прогонах"
        )

    # Проверка калибровки: систематическое перепредсказание важнее f1.
    calibration = _calibration(runs)
    if calibration:
        payload["calibration"] = calibration
        for item in calibration:
            findings.append(
                f"{item['model']}: доля предсказанных положительных "
                f"{item['predicted_rate']:.1%} против фактических "
                f"{item['actual_rate']:.1%} — "
                + ("переоценка" if item["ratio"] > 1 else "недооценка")
            )

    return findings


def _calibration(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Насколько систематично модель переоценивает долю положительного класса."""
    by_model: dict[str, list[tuple[float, float]]] = {}
    for run in runs:
        metrics = run["metrics"] or {}
        predicted = metrics.get("predicted_positive_rate")
        actual = metrics.get("actual_positive_rate")
        if predicted is None or actual is None or actual <= 0:
            continue
        by_model.setdefault(run["model"], []).append(
            (float(predicted), float(actual))
        )

    result: list[dict[str, Any]] = []
    for model, pairs in sorted(by_model.items()):
        predicted = float(np.mean([p for p, _ in pairs]))
        actual = float(np.mean([a for _, a in pairs]))
        result.append({
            "model": model,
            "n": len(pairs),
            "predicted_rate": round(predicted, 4),
            "actual_rate": round(actual, 4),
            "ratio": round(predicted / actual, 2) if actual > 0 else None,
        })
    return result
