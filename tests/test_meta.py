"""Проверки Meta Learning (7.b.iii).

Главная проверка здесь — не «числа посчитались», а «выводы не выдуманы».
Соблазн анализа в том, что он охотно производит правдоподобные значения
«важности» и для признаков, которые никогда не менялись. Такие цифры
выглядят как результат и ничего не значат, поэтому основная работа
модуля состоит в том, чтобы честно сказать: оценить не на чем.
"""

from __future__ import annotations

import json

import pytest

from src.meta import (
    MIN_CORRELATION,
    analyse,
    collect_runs,
    condition_influence,
    dynamics,
    feature_influence,
    spearman,
)


# ---------------------------------------------------------------------------
# Подготовка
# ---------------------------------------------------------------------------


def manifest(index: int, f1: float, **settings) -> dict:
    """Манифест батча с одной моделью и заданными настройками."""
    return {
        "batch_idx": index,
        "metrics": {
            "lr": {
                "f1": f1,
                "roc_auc": 0.7,
                "precision": f1,
                "recall": f1,
                "predicted_positive_rate": 0.3,
                "actual_positive_rate": 0.05,
            }
        },
        "hyperparameters": {"lr": {"training_mode": "scratch", **settings}},
        "data": {"rows_clean": 1000 + index, "target": {"positive_rate": 0.1 - index * 0.001}},
        "preprocessor": {"n_features": 300},
        "drift": {"status": "ok", "max_psi": 0.1},
    }


@pytest.fixture
def varied() -> list[dict]:
    """Восемь батчей: половина обучается с нуля, половина — дообчением."""
    return [
        manifest(i, 0.10 if i < 4 else 0.30, training_mode=("scratch" if i < 4 else "partial_fit"))
        for i in range(8)
    ]


# ---------------------------------------------------------------------------
# Статистика
# ---------------------------------------------------------------------------


def test_spearman_detects_monotonic_relation():
    """Идеальная монотонная связь даёт ρ = 1 (или −1)."""
    assert spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]) == pytest.approx(-1.0)


def test_spearman_handles_ties():
    """Ранги с учётом равных значений: корреляция считается, а не падает.

    Достигнутое число итераций у `mlp` повторяется почти на каждом
    батче, то есть серии с обилием совпадений — норма, а не аномалия.
    """
    value = spearman([1, 1, 1, 2, 2, 3], [1, 2, 3, 4, 5, 6])
    assert value is not None
    assert 0.0 < value < 1.0


def test_spearman_returns_none_on_constant_input():
    """Постоянный признак даёт `None`, а не нулевую корреляцию.

    Ноль — это результат («связи нет»), `None` — невозможность вычислить.
    Смешивать их нельзя: иначе в вывод попадёт «влияния нет» там, где
    данных для проверки просто не было.
    """
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None
    assert spearman([1, 2, 3, 4], [5, 5, 5, 5]) is None


def test_spearman_returns_none_on_short_series():
    """На двух точках корреляция не вычисляется."""
    assert spearman([1, 2], [1, 2]) is None


# ---------------------------------------------------------------------------
# Влияние настроек
# ---------------------------------------------------------------------------


def test_varied_feature_is_ranked(varied):
    """Признак, менявшийся внутри модели, оценивается и ранжируется."""
    runs = collect_runs(varied)
    result = feature_influence(runs, "training_mode")

    assert result["verdict"] == "varied"
    assert result["n_distinct"] == 2
    assert result["scope"] == "внутри семейства моделей"

    top = result["ranked"][0]
    assert top["model"] == "lr"
    assert top["feature"] == "training_mode"
    assert top["best_value"] == "partial_fit"
    assert top["spread"] == pytest.approx(0.20, abs=1e-6)
    assert top["groups"]["partial_fit"]["mean"] > top["groups"]["scratch"]["mean"]


def test_features_are_compared_only_within_one_model():
    """Значение, совпадающее с границей между моделями, сравнивать нельзя.

    У случайного леса `max_depth=12`, у дерева — `10`. Сравнив эти
    группы напрямую, анализ измерил бы разницу между лесом и деревом и
    выдал бы её за «влияние глубины»: вывод правдоподобный, бессмысленный
    и неверный.
    """
    runs = collect_runs([
        {"batch_idx": i,
         "metrics": {"rf": {"f1": 0.30}},
         "hyperparameters": {"rf": {"max_depth": 12}},
         "data": {}, "preprocessor": {}, "drift": {}}
        for i in range(5)
    ] + [
        {"batch_idx": 10 + i,
         "metrics": {"dt": {"f1": 0.15}},
         "hyperparameters": {"dt": {"max_depth": 10}},
         "data": {}, "preprocessor": {}, "drift": {}}
        for i in range(5)
    ])
    result = feature_influence(runs, "max_depth")

    assert result["verdict"] == "confounded"
    assert "между моделями" in result["note"]
    assert "ranked" not in result
    # Обе модели указаны как несравнённые: ни у одной нет двух значений.
    assert sorted(result["not_compared"]) == ["dt", "rf"]


def test_constant_feature_declares_itself_unmeasurable(varied):
    """Постоянная настройка честно объявляется непроверяемой.

    Это главное требование к модулю: посчитать «важность» признака,
    который не менялся, — значит выдать за результат цифру без смысла.
    """
    for item in varied:
        item["hyperparameters"]["lr"]["max_depth"] = 3
    runs = collect_runs(varied)
    result = feature_influence(runs, "max_depth")

    assert result["verdict"] == "constant"
    assert result["value"] == 3
    assert "не оценивается" in result["note"]
    assert "groups" not in result


def test_small_groups_are_not_a_result():
    """Группа из одного наблюдения не даёт права на вывод."""
    manifests = [
        manifest(0, 0.1, training_mode="a"),
        manifest(1, 0.1, training_mode="a"),
        manifest(2, 0.1, training_mode="a"),
        manifest(3, 0.9, training_mode="b"),
    ]
    runs = collect_runs(manifests)
    result = feature_influence(runs, "training_mode")

    assert result["verdict"] == "insufficient"
    # Причина должна называть конкретную группу, а не констатировать
    # общую нехватку: по названию видно, что именно пересобрать.
    assert "b" in str(result["comparisons"])


def test_missing_feature_is_reported_as_no_data(varied):
    """Отсутствующий признак — «нет данных», а не нулевое влияние."""
    runs = collect_runs(varied)
    assert feature_influence(runs, "absent")["verdict"] == "no_data"


# ---------------------------------------------------------------------------
# Влияние условий обучения
# ---------------------------------------------------------------------------


def test_correlation_needs_enough_observations(varied):
    """Корреляция не считается на нехватке наблюдений."""
    short = varied[:3]
    results = condition_influence(collect_runs(short))
    assert all(item["n"] >= MIN_CORRELATION for item in results)
    assert results == []


def test_correlation_is_computed_on_long_series(varied):
    """На достаточной серии корреляция считается и ранжируется."""
    runs = collect_runs(varied)
    for item in runs:
        item["settings"]["rows_used"] = 1000 + item["batch_idx"] * 10
    results = condition_influence(runs)

    by_feature = {item["feature"]: item for item in results}
    assert "rows_used" in by_feature
    assert by_feature["rows_used"]["verdict"] == "computed"
    # В фикстуре f1 ступенчатый (0.10 на первых четырёх батчах, 0.30 на
    # остальных), а объём строк растёт монотонно, поэтому связь сильная,
    # но не идеальная — именно такое значение и должно получиться.
    assert by_feature["rows_used"]["spearman"] > 0.8


def test_correlation_is_one_for_perfect_monotone_series():
    """Идеально монотонная связь даёт ρ = 1."""
    runs = collect_runs([manifest(i, 0.1 + i * 0.01) for i in range(10)])
    for item in runs:
        item["settings"]["rows_used"] = 1000 + item["batch_idx"] * 10
    results = condition_influence(runs)

    by_feature = {item["feature"]: item for item in results}
    assert by_feature["rows_used"]["spearman"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Динамика
# ---------------------------------------------------------------------------


def test_dynamics_reports_trend_and_best(varied):
    """Динамика содержит лучший батч, первый, последний и направление."""
    result = dynamics(varied)
    model = result["models"]["lr"]

    assert model["n_batches"] == 8
    assert model["best"]["f1"] == 0.30
    assert model["trend"] == "растёт"
    assert result["wins"] == {"lr": 8}


def test_dynamics_winner_is_computed_per_batch():
    """Победитель по батчу — тот, у кого выше f1 именно на этом батче."""
    manifests = [
        {"batch_idx": 0, "metrics": {"lr": {"f1": 0.1}, "mlp": {"f1": 0.4}},
         "data": {}, "preprocessor": {}, "drift": {}},
        {"batch_idx": 1, "metrics": {"lr": {"f1": 0.5}, "mlp": {"f1": 0.2}},
         "data": {}, "preprocessor": {}, "drift": {}},
    ]
    result = dynamics(manifests)

    assert result["wins"] == {"mlp": 1, "lr": 1}
    assert result["best_by_batch"] == [
        {"batch_idx": 0, "model": "mlp"},
        {"batch_idx": 1, "model": "lr"},
    ]


def test_dynamics_without_manifests():
    """Пустой список манифестов не приводит к исключению."""
    result = dynamics([])
    assert result["models"] == {}
    assert result["wins"] == {}


# ---------------------------------------------------------------------------
# Итоговый анализ
# ---------------------------------------------------------------------------


def test_analysis_refuses_to_conclude_on_few_batches():
    """На малом числе батчей выводов нет, и это сказано прямо."""
    payload = analyse([manifest(0, 0.1), manifest(1, 0.2)], min_runs=3)

    assert payload["verdict"] == "insufficient"
    assert "голословными" in payload["note"]
    assert "findings" not in payload


def test_analysis_lists_constant_settings(varied):
    """Неизменявшиеся настройки перечислены, а не выкинуты молча.

    Читатель должен видеть, что́ именно осталось непроверенным, иначе
    «проверенного влияния нет» будет выглядеть как «влияния нет».
    """
    for item in varied:
        item["hyperparameters"]["lr"]["learning_rate"] = 0.01
    payload = analyse(varied, min_runs=3)

    constant = payload["settings_influence"]["constant"]
    names = {item["feature"] for item in constant}
    assert "learning_rate" in names
    assert "n_distinct" not in constant[0]


def test_analysis_reports_calibration_problem(varied):
    """Завышение доли положительного класса попадает в выводы.

    Это и есть причина, по которой в реальном прогоне понадобился
    пороговый слой: без явного вывода переоценка выглядит как
    «модель работает».
    """
    payload = analyse(varied, min_runs=3)

    assert payload["calibration"][0]["predicted_rate"] == pytest.approx(0.3)
    assert payload["calibration"][0]["actual_rate"] == pytest.approx(0.05)
    assert any("переоценка" in finding for finding in payload["findings"])


def test_analysis_is_json_serialisable(varied):
    """Результат уходит в JSON-артефакт, значит быть должен сериализуем."""
    payload = analyse(varied, min_runs=3)
    text = json.dumps(payload, ensure_ascii=False)
    assert "findings" in json.loads(text)


def test_analysis_sorts_manifests_itself():
    """Порядок манифестов на входе не влияет на результат."""
    straight = analyse([manifest(i, 0.1 * (i + 1)) for i in range(6)], min_runs=3)
    shuffled = analyse(
        [manifest(i, 0.1 * (i + 1)) for i in reversed(range(6))], min_runs=3
    )
    assert straight["n_batches"] == shuffled["n_batches"]
    assert straight["dynamics"]["models"] == shuffled["dynamics"]["models"]


def test_manifests_without_hyperparameters_do_not_crash():
    """Манифесты, написанные до появления настроек, не ломают анализ.

    Так бывает при переходе на новую версию кода на уже накопленных
    артефактах: старые батчи не содержат настроек, и это не причина
    отказывать в анализе целиком. Отдельно помечается, что настроек
    нет, иначе пробел в выводах выглядел бы как «влияния нет».
    """
    manifests = [
        {"batch_idx": i, "metrics": {"lr": {"f1": 0.2}},
         "data": {}, "preprocessor": {}, "drift": {}}
        for i in range(6)
    ]
    payload = analyse(manifests, min_runs=3)

    assert payload["verdict"] == "ok"
    assert payload["settings_recorded"] is False
    assert payload["settings_influence"]["varied"] == []


def test_settings_recorded_is_reported_when_present(varied):
    """Наличие настроек в манифестах фиксируется явно."""
    payload = analyse(varied, min_runs=3)
    assert payload["settings_recorded"] is True
