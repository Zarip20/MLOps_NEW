"""Проверки правил качества данных.

Тесты фиксируют поведение, которое было исправлено в ходе baseline-прогона:
разделение пропусков и нарушений (D-27), условные правила (D-1) и
нормировку метрики динамических правил (D-5).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_quality import DataQualityEvaluator
from src.rules import RuleSet, clean_by_rules, evaluate_rules


# ---------------------------------------------------------------------------
# Пропуск — это не нарушение (D-27)
# ---------------------------------------------------------------------------


def test_missing_values_are_not_violations_when_ignored(rule_set, sample_batch):
    """Правило с on_missing=ignore не должно считать пропуск нарушением.

    Именно из-за смешения этих двух вещей валидатор объявлял невалидными
    заведомо корректные колонки, а очистка удаляла пропуски.
    """
    premium = rule_set.rules[0]
    assert premium.on_missing == "ignore"

    frame = pd.DataFrame({"PREMIUM": [100.0, None, 200.0]})
    violations = premium.violation_mask(frame)
    assert violations.tolist() == [False, False, False]


def test_missing_is_violation_when_configured(rule_set, sample_batch):
    """Тот же пропуск при on_missing=violation является нарушением."""
    cargo = next(r for r in rule_set.rules if r.name == "cargo_capacity_required")
    assert cargo.on_missing == "violation"

    frame = pd.DataFrame(
        {"CARRYING_CAPACITY": [1500.0, None, 0.0, 800.0],
         "TYPE_VEHICLE": ["Truck", "Truck", "Truck", "Truck"]}
    )
    violations = cargo.violation_mask(frame)
    assert violations.tolist() == [False, True, True, False]


def test_validity_excludes_missing_from_checked(rule_set, sample_batch):
    """В `validity` пропуски не должны попадать в число проверенных значений."""
    evaluator = DataQualityEvaluator(rule_set, feature_cols=list(sample_batch.columns))
    result = evaluator.validity(sample_batch)

    # SEATS_NUM: 3 пропуска из 20 -> проверено 17
    assert result["seats_non_negative"]["checked"] == 17
    assert result["seats_non_negative"]["violations"] == 0


def test_validity_does_not_flag_columns_with_only_missing(rule_set):
    """Колонка целиком из пропусков не считается невалидной."""
    frame = pd.DataFrame({"PREMIUM": [None, None, None]})
    premium = RuleSet.from_config(
        [{"name": "p", "column": "PREMIUM", "op": "ge", "value": 0}]
    ).rules[0]
    assert premium.violation_mask(frame).sum() == 0


# ---------------------------------------------------------------------------
# Сентинельные нули (решение Р-2)
# ---------------------------------------------------------------------------


def test_zero_insured_value_is_valid(rule_set, sample_batch):
    """`INSURED_VALUE == 0` — значимое значение, а не нарушение."""
    rule = next(r for r in rule_set.rules if r.name == "insured_value_non_negative")
    assert not rule.violation_mask(sample_batch).any()


# ---------------------------------------------------------------------------
# Условные правила (D-1)
# ---------------------------------------------------------------------------


def test_conditional_rule_ignores_non_cargo_vehicles(rule_set, sample_batch):
    """Мотоцикл с пустой грузоподъёмностью не нарушает правило для грузовых."""
    rule = next(r for r in rule_set.rules if r.name == "cargo_capacity_required")
    motorcycles = sample_batch[sample_batch["TYPE_VEHICLE"] == "Motor-cycle"]
    assert len(motorcycles) == 4
    assert rule.violation_mask(motorcycles).sum() == 0


def test_conditional_rule_catches_all_cargo_violations(rule_set, sample_batch):
    """Пустая и нулевая грузоподъёмность у грузовика — оба случая нарушение."""
    rule = next(r for r in rule_set.rules if r.name == "cargo_capacity_required")
    # 4 с пустой + 3 с нулевой
    assert rule.violation_mask(sample_batch).sum() == 7


def test_rule_case_sensitivity_is_configurable(rule_set):
    """Регистр значения задаётся конфигурацией, а не предполагается кодом.

    Исходный дефект D-1 был ровно в этом: в коде стояло 'truck',
    в данных — 'Truck'.
    """
    wrong = RuleSet.from_config(
        [{
            "name": "cargo", "column": "CARRYING_CAPACITY", "op": "gt", "value": 0,
            "on_missing": "violation",
            "applies_when": {"column": "TYPE_VEHICLE", "op": "in", "value": ["truck"]},
        }]
    ).rules[0]
    frame = pd.DataFrame(
        {"CARRYING_CAPACITY": [None], "TYPE_VEHICLE": ["Truck"]}
    )
    assert wrong.violation_mask(frame).sum() == 0  # не совпало — правило мертво

    right = RuleSet.from_config(
        [{
            "name": "cargo", "column": "CARRYING_CAPACITY", "op": "gt", "value": 0,
            "on_missing": "violation",
            "applies_when": {"column": "TYPE_VEHICLE", "op": "in", "value": ["Truck"]},
        }]
    ).rules[0]
    assert right.violation_mask(frame).sum() == 1  # совпало — правило работает


# ---------------------------------------------------------------------------
# Метрика динамических правил (D-5)
# ---------------------------------------------------------------------------


def test_dynamic_violation_ratio_is_normalised_by_antecedent(rule_set, sample_batch, association_rules):
    """Доля нарушений считается от строк с антецендентом, а не от батча."""
    evaluator = DataQualityEvaluator(rule_set)
    result = evaluator.check_dynamic_rules(sample_batch, association_rules, min_antecedent_rows=1)

    # 3 строки с TYPE_VEHICLE = 'Special construction' и 3 без совпадения
    entry = result["TYPE_VEHICLE = Special construction -> USAGE = Special Construction"]
    assert entry["antecedent_rows"] == 3
    # USAGE = 'Special Construction' у всех трёх -> нарушений 0
    assert entry["violations"] == 0
    assert entry["violation_ratio"] == 0.0
    # Поддержка считается от размера батча
    assert entry["support"] == pytest.approx(3 / len(sample_batch))


def test_dynamic_ratio_detects_real_violation(rule_set, association_rules):
    """Нарушение фиксируется, когда антецендент есть, а следствия нет."""
    frame = pd.DataFrame(
        {
            "TYPE_VEHICLE": ["Special construction"] * 8 + ["Motor-cycle"] * 2,
            "USAGE": ["Own Goods"] * 6 + ["Special Construction"] * 2 + ["Private"] * 2,
        }
    )
    evaluator = DataQualityEvaluator(rule_set)
    result = evaluator.check_dynamic_rules(
        frame, association_rules[:1], min_antecedent_rows=1
    )
    entry = result["TYPE_VEHICLE = Special construction -> USAGE = Special Construction"]
    # 8 строк под антецендентом, 6 из них без следствия -> 0.75
    assert entry["antecedent_rows"] == 8
    assert entry["violations"] == 6
    assert entry["violation_ratio"] == pytest.approx(0.75)


def test_dynamic_ratio_suppressed_for_tiny_support(rule_set, sample_batch, association_rules):
    """Малый антецендент не должен давать «100 % нарушений»."""
    evaluator = DataQualityEvaluator(rule_set)
    result = evaluator.check_dynamic_rules(sample_batch, association_rules, min_antecedent_rows=30)
    for entry in result.values():
        assert entry["violation_ratio"] is None
        assert entry["sufficient_support"] is False
        # Исходная доля сохраняется для диагностики
        assert "raw_violation_ratio" in entry


def test_dynamic_rule_absent_in_batch(rule_set, sample_batch, association_rules):
    """Отсутствие антецендента — не ошибка, а нулевая поддержка."""
    frame = sample_batch.copy()
    frame["MAKE"] = "NOBODY"
    evaluator = DataQualityEvaluator(rule_set)
    result = evaluator.check_dynamic_rules(frame, association_rules, min_antecedent_rows=1)
    entry = result["MAKE = ISUZU -> TYPE_VEHICLE = Truck"]
    assert entry["antecedent_rows"] == 0
    assert entry["violation_ratio"] is None


# ---------------------------------------------------------------------------
# Очистка
# ---------------------------------------------------------------------------


def test_clean_data_removes_rule_violations(rule_set, sample_batch):
    """Очистка удаляет строки, нарушающие правило с severity=error."""
    # Порог нарушений поднят намеренно: в 20-строчном батче доля 7/20 = 0.35
    # превысила бы боевой порог 0.1, и правило было бы пропущено как
    # настроенное неверно. Поведение при превышении порога проверяется
    # отдельно — в test_rule_skipped_when_violation_ratio_too_high.
    evaluator = DataQualityEvaluator(
        rule_set, max_violation_ratio=0.5, feature_cols=list(sample_batch.columns)
    )
    cleaned, report = evaluator.clean_data(sample_batch)

    assert report["rows_in"] == len(sample_batch)
    assert report["rows_removed"] == report["rows_in"] - report["rows_out"]
    assert len(cleaned) == report["rows_out"]
    assert cleaned["TYPE_VEHICLE"].str.contains("Motor-cycle").any()
    assert not (
        (cleaned["TYPE_VEHICLE"].isin(["Truck", "Tanker"]))
        & (cleaned["CARRYING_CAPACITY"].fillna(0) <= 0)
    ).any()


def test_clean_data_totals_are_consistent(rule_set, sample_batch):
    """Сумма удалений по правилам сходится с общим числом удалённых строк."""
    evaluator = DataQualityEvaluator(
        rule_set, max_violation_ratio=0.5, feature_cols=list(sample_batch.columns)
    )
    _, report = evaluator.clean_data(sample_batch)

    by_rules = report.get("rows_removed_by_rules", 0)
    by_other = report["dropped_by_missing"] + report["dropped_duplicates"]
    assert by_rules + by_other == report["rows_removed"]


def test_warn_rules_do_not_remove_rows(rule_set, sample_batch):
    """Правила с severity=warn только наблюдаются, но не чистят."""
    assert any(not rule.is_error for rule in rule_set)
    evaluator = DataQualityEvaluator(
        rule_set, max_violation_ratio=0.5, feature_cols=list(sample_batch.columns)
    )
    cleaned, report = evaluator.clean_data(sample_batch)

    applied = {entry["rule"] for entry in report["applied"]}
    assert "make_watch" not in applied
    assert len(cleaned) > 0


def test_rule_skipped_when_violation_ratio_too_high(rule_set):
    """Правило, нарушаемое слишком часто, не применяется и попадает в отчёт."""
    frame = pd.DataFrame(
        {
            "CARRYING_CAPACITY": [None] * 100,
            "TYPE_VEHICLE": ["Truck"] * 100,
        }
    )
    result, report = clean_by_rules(
        frame,
        RuleSet.from_config(
            [{
                "name": "cargo", "column": "CARRYING_CAPACITY", "op": "gt", "value": 0,
                "on_missing": "violation",
                "applies_when": {"column": "TYPE_VEHICLE", "op": "in", "value": ["Truck"]},
            }]
        ),
        max_violation_ratio=0.1,
    )
    assert len(result) == 100, "строки не должны удаляться"
    assert report["skipped"], "пропуск правила должен быть зафиксирован"
    assert "превышает порог" in report["skipped"][0]["reason"]


def test_rule_missing_column_is_reported_not_fatal(rule_set, sample_batch):
    """Правило по отсутствующей колонке не роняет конвейер, а отмечается."""
    frame = sample_batch.drop(columns=["CARRYING_CAPACITY"])
    evaluator = DataQualityEvaluator(
        rule_set, max_violation_ratio=0.5, feature_cols=list(sample_batch.columns)
    )
    cleaned, report = evaluator.clean_data(frame)

    # Ничего не удалено: единственное правило по отсутствующей колонке
    # пропущено, остальные правила на этих данных не нарушаются.
    assert len(cleaned) == len(frame)
    assert report["rows_removed"] == 0
    skipped = {entry["rule"]: entry for entry in report["skipped"]}
    assert "cargo_capacity_required" in skipped
    assert "колонка отсутствует" in skipped["cargo_capacity_required"]["reason"]
    assert evaluator.skipped_rules(frame) == ["cargo_capacity_required"]


# ---------------------------------------------------------------------------
# Прочие метапараметры
# ---------------------------------------------------------------------------


def test_completeness_ignores_target_column(rule_set, sample_batch):
    """Полнота считается по признакам и не искажается пустой целевой колонкой."""
    evaluator = DataQualityEvaluator(
        rule_set, feature_cols=["SEATS_NUM", "CARRYING_CAPACITY", "PREMIUM"]
    )
    result = evaluator.completeness(sample_batch)

    assert result["n_columns"] == 3
    # CARRYING_CAPACITY пуста в 8 строках из 20: 4 грузовика без
    # грузоподъёмности и 4 мотоцикла
    assert result["max_col_missing_ratio"] == pytest.approx(8 / 20)
    assert result["max_col_missing_ratio_column"] == "CARRYING_CAPACITY"
    # CLAIM_PAID пуста в 11 строках, но в метрику не входит: иначе
    # «полнота» отражала бы полноту метки, а не признаков
    assert result["total_missing_ratio"] < 0.5
    assert "CLAIM_PAID" not in result


def test_type_check_detects_non_numeric(rule_set):
    """Нечисловой мусор в числовой колонке обнаруживается (D-26)."""
    frame = pd.DataFrame({"EFFECTIVE_YR": ["08", "1C", "04", "4A", None]})
    evaluator = DataQualityEvaluator(
        rule_set, expected_numeric=["EFFECTIVE_YR", "PROD_YEAR"]
    )
    result = evaluator.type_check(frame)

    assert "EFFECTIVE_YR" in result
    assert result["EFFECTIVE_YR"]["non_numeric"] == 2
    assert set(result["EFFECTIVE_YR"]["examples"]) == {"1C", "4A"}
    assert "PROD_YEAR" not in result


def test_timeliness_reports_span_and_gaps(rule_set, sample_batch):
    """Своевременность возвращает границы и разрывы между датами."""
    evaluator = DataQualityEvaluator(rule_set)
    result = evaluator.timeliness(sample_batch, "INSR_BEGIN", date_format="%d-%b-%y")

    assert result["available"] is True
    assert result["n_unique_dates"] == 1
    assert result["max_gap_days"] == 0
    assert result["unparsed_dates"] == 0
    assert result["min_date"].startswith("2015-08-08")
