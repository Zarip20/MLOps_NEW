"""Проверки устойчивого инференса (6.b.ii).

Требование: «валидация входных данных, fallback на безопасную модель при
аномалиях или редких категориях, деградация вместо падения». Проверяются
все три части, и главное — что деградация **видна**: прогноз, полученный
не той моделью, на которую рассчитывали, обязан быть помечен. Иначе
пользователь примет аварийный ответ за полноценный.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inference import (
    STATUS_DEGRADED,
    STATUS_OK,
    STATUS_REJECTED,
    UNKNOWN_CATEGORY,
    ValidationReport,
    SafePredictor,
    choose_threshold,
    count_outliers,
    forget_training_ranges,
    known_ranges,
    remember_training_ranges,
    validate_input,
)

NUMERICAL = ["INSURED_VALUE", "PREMIUM"]
CATEGORICAL = ["MAKE", "USAGE"]


def make_frame(rows: int = 10, **overrides) -> pd.DataFrame:
    """Нормальный кадр со всеми признаками."""
    frame = pd.DataFrame({
        "INSURED_VALUE": [100000.0 + i for i in range(rows)],
        "PREMIUM": [500.0 + i for i in range(rows)],
        "MAKE": ["ISUZU"] * rows,
        "USAGE": ["Own Goods"] * rows,
    })
    for column, value in overrides.items():
        frame[column] = value
    return frame


REQUIRED = NUMERICAL + CATEGORICAL


# ---------------------------------------------------------------------------
# Проверка входа
# ---------------------------------------------------------------------------


def test_valid_input_passes():
    """Полный корректный файл принимается без замечаний."""
    report = validate_input(make_frame(), REQUIRED, NUMERICAL)

    assert report.status == STATUS_OK
    assert report.accepted
    assert report.problems == []


def test_missing_column_is_rejected():
    """Без обязательной колонки прогноз бессмыслен — вход отвергается."""
    frame = make_frame().drop(columns=["PREMIUM"])
    report = validate_input(frame, REQUIRED, NUMERICAL)

    assert report.status == STATUS_REJECTED
    assert not report.accepted
    assert "PREMIUM" in report.problems[0]


def test_empty_file_is_rejected():
    """Пустой файл отвергается, а не превращается в пустой результат."""
    report = validate_input(make_frame(0), REQUIRED, NUMERICAL)

    assert report.status == STATUS_REJECTED
    assert "строк" in report.problems[0]


def test_non_numeric_values_are_repaired_not_rejected():
    """Мусор в числовой колонке чинится, а не отвергает весь файл.

    Иначе файл с одной испорченной ячейкой потерял бы тысячу строк
    нормальных данных.
    """
    frame = make_frame()
    frame["PREMIUM"] = ["500"] * 9 + ["не число"]
    report = validate_input(frame, REQUIRED, NUMERICAL)

    assert report.status == STATUS_DEGRADED
    assert report.accepted, "исправимое не должно отвергаться"
    assert any("числовые" in item for item in report.repairs)
    assert pd.isna(frame["PREMIUM"].iloc[-1])


def test_unknown_category_is_named_not_silently_replaced():
    """Незнакомая категория перечисляется, а не подменяется.

    One-hot-кодировщик не обучен на `__unknown__` и закодировал бы её в
    те же нули, что и исходное значение. Подмена не изменила бы ни
    одного числа в прогнозе, а создала бы видимость починки.
    """
    frame = make_frame()
    frame.loc[0, "MAKE"] = "ВЫДУМАННАЯ МАРКА"
    report = validate_input(
        frame, REQUIRED, NUMERICAL, known_categories={"MAKE": {"ISUZU"}}
    )

    assert report.status == STATUS_DEGRADED
    assert frame.loc[0, "MAKE"] == "ВЫДУМАННАЯ МАРКА", "значение не должно меняться"
    assert any("категор" in item for item in report.problems)
    assert "ВЫДУМАННАЯ МАРКА" in report.problems[0]
    # И сказано, что исправить без переобучения нельзя
    assert "переобучения" in report.problems[0]


def test_known_categories_pass_untouched():
    """Знакомые категории не трогаются."""
    frame = make_frame()
    before = frame["MAKE"].tolist()
    report = validate_input(
        frame, REQUIRED, NUMERICAL, known_categories={"MAKE": {"ISUZU"}}
    )
    assert frame["MAKE"].tolist() == before
    assert report.status == STATUS_OK


def test_many_outliers_are_reported():
    """Файл, в котором почти всё выходит за обучающий диапазон, отвергается."""
    training = make_frame(100)
    ranges = remember_training_ranges(training, NUMERICAL)
    frame = make_frame(10, INSURED_VALUE=[10**9] * 10)
    report = validate_input(
        frame, REQUIRED, NUMERICAL, ranges=ranges, max_outlier_ratio=0.2
    )

    assert report.status == STATUS_REJECTED
    assert any("выброс" in item for item in report.problems)


def test_report_is_serialisable():
    """Отчёт уходит в JSON, значит должен быть сериализуем."""
    report = validate_input(make_frame(), REQUIRED, NUMERICAL)
    payload = report.as_dict()
    assert payload["accepted"] is True
    assert isinstance(payload["checks"], list)


# ---------------------------------------------------------------------------
# Калибровка порога
# ---------------------------------------------------------------------------


def test_threshold_hits_the_target_rate():
    """Порог по квантилю даёт заданную долю положительных решений."""
    probabilities = np.linspace(0.01, 0.99, 1000)
    info = choose_threshold(probabilities, 0.05)

    assert info["achieved_rate"] == pytest.approx(0.05, abs=0.005)
    assert 0.7 < info["threshold"] < 0.99


def test_threshold_reports_what_would_happen_without_it():
    """Видно, что дал бы обычный порог 0,5 — иначе выигрыш не проверить."""
    probabilities = np.linspace(0.6, 0.99, 1000)
    info = choose_threshold(probabilities, 0.05)

    assert info["raw_rate_at_half"] == pytest.approx(1.0)
    assert info["achieved_rate"] == pytest.approx(0.05, abs=0.005)


def test_threshold_on_empty_input_is_explicit():
    """На пустом входе порог не выдумывается, а объясняется."""
    info = choose_threshold(np.array([]), 0.05)

    assert info["threshold"] == 0.5
    assert info["achieved_rate"] is None
    assert "нет вероятностей" in info["reason"]


def test_threshold_ignores_non_finite_values():
    """Нечисловые вероятности не ломают выбор порога."""
    probabilities = np.array([0.1, 0.5, 0.9, np.nan, np.inf])
    info = choose_threshold(probabilities, 0.4)

    assert np.isfinite(info["threshold"])


# ---------------------------------------------------------------------------
# Деградация
# ---------------------------------------------------------------------------


class FakePreprocessor:
    """Препроцессор, который умеет и не умеет.transform."""

    def __init__(self, works: bool = True) -> None:
        self.works = works
        self.seen: list | None = None

    def transform(self, frame: pd.DataFrame):
        if not self.works:
            raise ValueError("схема признаков не совпадает")
        self.seen = list(frame.columns)
        return np.zeros((len(frame), 2))


class FakeModel:
    """Модель с заданной вероятностью положительного класса."""

    def __init__(self, value: float = 0.9, fail: bool = False) -> None:
        self.value = value
        self.fail = fail

    def predict_proba(self, matrix):
        if self.fail:
            raise RuntimeError("модель не работает")
        rows = matrix.shape[0]
        return np.tile([1.0 - self.value, self.value], (rows, 1))

    def predict(self, matrix):
        return (self.predict_proba(matrix)[:, 1] >= 0.5).astype(int)


def test_fallback_used_when_best_model_fails():
    """Продуктовая модель сломалась — прогноз даёт запасная, а не отказ."""
    predictor = SafePredictor(
        {"best": FakeModel(0.9, fail=True), "lr": FakeModel(0.6)},
        FakePreprocessor(),
        REQUIRED,
        fallback="lr",
    )
    result = predictor.predict(make_frame(), ValidationReport())

    assert result.model == "lr"
    assert result.fallback_used is True
    assert any("запасная модель" in note for note in result.notes)


def test_preprocessor_failure_degrades_to_constant():
    """Если препроцессор нерабочий, выдаётся константа, а не исключение.

    Пустой файл результата непригоден ни для чего, а ответ «страховых
    случаев нет» на месяц без случаев хотя бы пригоден для сверки.
    """
    predictor = SafePredictor(
        {"best": FakeModel(0.9)},
        FakePreprocessor(works=False),
        REQUIRED,
    )
    result = predictor.predict(make_frame(5), ValidationReport())

    assert result.model == "constant"
    assert result.predictions.sum() == 0
    assert any("константный прогноз" in note for note in result.notes)


def test_no_models_at_all_still_returns_a_result():
    """Отсутствие моделей не приводит к исключению."""
    predictor = SafePredictor({}, FakePreprocessor(), REQUIRED)
    result = predictor.predict(make_frame(3), ValidationReport())

    assert result.model == "constant"
    assert len(result.predictions) == 3


def test_degradation_is_visible_in_the_report():
    """Аварийный прогноз помечен: пользователь не примет его за полноценный."""
    predictor = SafePredictor(
        {"best": FakeModel(0.9, fail=True), "lr": FakeModel(0.6)},
        FakePreprocessor(),
        REQUIRED,
        fallback="lr",
    )
    result = predictor.predict(make_frame(), ValidationReport())

    payload = result.as_dict()
    assert payload["fallback_used"] is True
    assert payload["notes"]


def test_calibrated_threshold_applied_by_predictor():
    """Прогноз идёт по калиброванному порогу, а не по 0,5.

    Все вероятности заведомо выше 0,5: без калибровки положительными
    оказались бы все строки, с ней — ровно заданная доля.
    """
    class Spread:
        def predict_proba(self, matrix):
            rows = matrix.shape[0]
            positive = np.linspace(0.6, 0.99, rows)
            return np.column_stack([1.0 - positive, positive])

    predictor = SafePredictor(
        {"best": Spread()},
        FakePreprocessor(),
        REQUIRED,
        threshold_mode="quantile",
        target_positive_rate=0.1,
    )
    result = predictor.predict(make_frame(100), ValidationReport())

    assert result.predictions.sum() == 10
    assert result.threshold > 0.5


def test_ties_are_reported_not_hidden():
    """Одинаковые вероятности не дают достичь целевой доли — и это говорится.

    Квантиль при равных значениях равен им всем, и доля скачет до 1.
    Молчаливые «100 % положительных» выглядели бы как провал
    калибровки, хотя дело в свойстве самой модели.
    """
    predictor = SafePredictor(
        {"best": FakeModel(0.95)},
        FakePreprocessor(),
        REQUIRED,
        threshold_mode="quantile",
        target_positive_rate=0.1,
    )
    result = predictor.predict(make_frame(100), ValidationReport())

    assert result.predictions.sum() == 100
    assert any("различных" in note for note in result.notes)


def test_fixed_mode_keeps_plain_threshold():
    """Режим `fixed` оставляет обычный порог — для сравнения с калибровкой."""
    predictor = SafePredictor(
        {"best": FakeModel(0.95)},
        FakePreprocessor(),
        REQUIRED,
        threshold_mode="fixed",
        threshold=0.5,
        target_positive_rate=0.1,
    )
    result = predictor.predict(make_frame(100), ValidationReport())

    assert result.threshold == 0.5
    assert result.predictions.sum() == 100


def test_model_without_proba_falls_back_to_predict():
    """Модель без вероятностей даёт предсказания и без калибровки."""
    class NoProba:
        def predict(self, matrix):
            return np.ones(matrix.shape[0], dtype=int)

    predictor = SafePredictor(
        {"best": NoProba()}, FakePreprocessor(), REQUIRED, target_positive_rate=0.1
    )
    result = predictor.predict(make_frame(4), ValidationReport())

    assert result.probabilities is None
    assert result.threshold is None
    assert result.predictions.sum() == 4


# ---------------------------------------------------------------------------
# Обучающие диапазоны
# ---------------------------------------------------------------------------


def test_ranges_use_tukey_fence():
    """Границы строятся «забором Тьюки», устойчивым к жирным хвостам.

    `PREMIUM` распределён так, что квантили 0,5 % и 99,5 % разъезжаются
    на порядки; межквартильный забор втрое отсекает хвост, оставляя
    основную часть значений внутри.
    """
    values = list(range(1, 101)) + [10**6]
    ranges = remember_training_ranges(pd.DataFrame({"x": values}), ["x"])

    low, high = ranges["x"]
    assert low < 5
    assert 150 < high < 1000


def test_degenerate_spread_falls_back_to_quantiles():
    """При нулевом размахе забор выродился бы в точку — берутся квантили.

    Иначе у колонки из одинаковых значений любое отличие считалось бы
    выбросом, и нормальный файл отвергался бы целиком.
    """
    ranges = remember_training_ranges(
        pd.DataFrame({"x": [1.0] * 99 + [2.0]}), ["x"]
    )
    low, high = ranges["x"]

    assert low < high
    assert 0.9 <= low <= 1.0


def test_outlier_count_respects_ranges():
    """Считаются именно выходы за диапазон, а не любые отличия."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 100.0]})
    total = count_outliers(frame, ["x"], {"x": (0.0, 10.0)})

    assert total == 1


def test_ranges_do_not_leak_between_calls():
    """Проверка не зависит от того, что накопилось раньше.

    Глобальное состояние диапазонов делало бы результат проверки
    зависящим от порядка вызовов в процессе, а это в тестах выглядит
    как «иногда проходит, иногда нет».
    """
    frame = make_frame(10, INSURED_VALUE=[10**9] * 10)
    clean = make_frame(10)

    assert count_outliers(clean, NUMERICAL, {"INSURED_VALUE": (0.0, 1e12)}) == 0
    assert count_outliers(frame, NUMERICAL, {"INSURED_VALUE": (0.0, 1e12)}) == 0
    assert count_outliers(frame, NUMERICAL, {"INSURED_VALUE": (0.0, 1.0)}) == 10


def test_forget_training_ranges_clears_state():
    """Диапазоны можно сбросить при переходе к другому набору данных."""
    remember_training_ranges(make_frame(50), ["PREMIUM"])
    assert "PREMIUM" in known_ranges()

    forget_training_ranges()
    assert "PREMIUM" not in known_ranges()
