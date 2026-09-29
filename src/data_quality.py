"""Оценка качества данных и очистка.

Модуль переписан по результатам baseline-прогона (PLAN.md, §1.8). Основные
исправления:

* **D-27 — валидность больше не смешивается с полнотой.** Раньше проверка
  выглядела как «все значения >= 0», а на колонке с пропусками `NaN >= 0`
  даёт `False`, поэтому валидатор объявлял невалидными заведомо корректные
  колонки, а правила качества начинали считать пропуски нарушениями.
  Теперь пропуски исключаются из проверки валидности, а их судьба решается
  полем `on_missing` у каждого правила отдельно.
* **D-6 — валидность стала информативной.** Вместо булева «все значения
  неотрицательны» возвращаются число и доля нарушений вместе с числом
  проверенных значений.
* **D-26 — добавлена проверка типов.** В `EFFECTIVE_YR` обнаружено 580
  повреждённых значений (мусор вида `1C`, `4A`), которые прежний валидатор
  пропускал; теперь они учитываются как нарушения формата.
* **D-5 — доля нарушений динамических правил нормируется по антеценденту.**
  Раньше доля считалась от размера всего батча, поэтому любое правило с
  поддержкой 1 % давало нулевую метрику.
* **D-24 — метод переименован** в `check_static_rules`; ассоциативные правила
  проверяются отдельно, и в метаданных больше нет двух похожих ключей.
* **D-7 — пороги очистки перенесены в конфигурацию.**
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.rules import Rule, RuleSet, clean_by_rules, evaluate_rules
from src.utils import parse_datetime


class DataQualityEvaluator:
    """Расчёт метапараметров качества и очистка батча."""

    def __init__(
        self,
        rules: RuleSet,
        *,
        max_row_missing_ratio: float = 0.5,
        max_violation_ratio: float = 0.1,
        feature_cols: Sequence[str] | None = None,
        numeric_cols: Sequence[str] | None = None,
        expected_numeric: Sequence[str] | None = None,
    ) -> None:
        self.rules = rules
        self.max_row_missing_ratio = float(max_row_missing_ratio)
        self.max_violation_ratio = float(max_violation_ratio)
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        # Числовые колонки источника: шире, чем признаки, — нужны для
        # проверки формата значений в колонках, не вошедших в модель.
        self.numeric_cols = list(expected_numeric or numeric_cols or [])

    # ------------------------------------------------------------------
    # Полнота
    # ------------------------------------------------------------------

    def completeness(self, df: pd.DataFrame) -> dict[str, float]:
        """Полнота данных.

        Считается по колонкам признаков, а не по всему батчу: иначе метрику
        определяет целевая колонка `CLAIM_PAID`, пустая по построению
        (в baseline она давала `max_col_missing_ratio` = 0.89…0.94 вместо
        реальной полноты признаков).

        Returns:
            Доли пропусков: по всем ячейкам, в худшей колонке, в худшей строке
            и средняя по батчу.
        """
        columns = [column for column in (self.feature_cols or df.columns) if column in df.columns]
        if not columns:
            return {
                "total_missing_ratio": float("nan"),
                "max_col_missing_ratio": float("nan"),
                "max_row_missing_ratio": float("nan"),
                "mean_row_missing_ratio": float("nan"),
                "n_columns": 0,
            }

        frame = df[columns]
        rows = max(len(frame), 1)
        cells = max(len(frame) * len(columns), 1)

        per_column = frame.isna().sum()
        per_row = frame.isna().sum(axis=1)

        return {
            "total_missing_ratio": float(per_column.sum() / cells),
            "max_col_missing_ratio": float(per_column.max() / rows),
            "max_col_missing_ratio_column": str(per_column.idxmax()),
            "max_row_missing_ratio": float(per_row.max() / len(columns)),
            "mean_row_missing_ratio": float(per_row.mean() / len(columns)),
            "n_columns": len(columns),
        }

    # ------------------------------------------------------------------
    # Валидность
    # ------------------------------------------------------------------

    def validity(
        self,
        df: pd.DataFrame,
        rules: Sequence[Rule] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Проверка диапазонов значений без учёта пропусков.

        Каждое правило даёт три числа: сколько значений проверено, сколько
        нарушено и какая это доля. Пропуски в проверку не входят — их
        учитывает `completeness`, а для отдельных правил решение о пропуске
        принимает поле `on_missing`.
        """
        applicable = rules if rules is not None else list(self.rules)
        result: dict[str, dict[str, Any]] = {}

        for rule in applicable:
            if rule.column not in df.columns:
                continue

            scope = rule.applicable_mask(df) if rule.applies_when else pd.Series(
                True, index=df.index
            )
            values = df.loc[scope, rule.column]

            if rule.on_missing == "violation" and rule.applies_when is not None:
                # Пропуск здесь — самостоятельное нарушение, считаем вместе.
                holds = rule.holds_mask(df.loc[scope]).fillna(False).astype(bool)
                checked = int(len(values))
                violations = int((~holds).sum())
            else:
                present = values.dropna()
                checked = int(len(present))
                if checked == 0:
                    result[rule.name] = {
                        "checked": 0, "violations": 0, "violation_ratio": float("nan"),
                    }
                    continue
                holds = rule.holds_mask(df.loc[present.index])
                holds = holds.reindex(present.index).fillna(False).astype(bool)
                violations = int((~holds).sum())

            result[rule.name] = {
                "checked": checked,
                "violations": violations,
                "violation_ratio": float(violations / checked) if checked else float("nan"),
            }

        return result

    def type_check(self, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Проверка того, что числовые колонки действительно числовые.

        Проверяются не только признаки, но и все колонки источника,
        объявленные числовыми в `schema.expected_numeric`. Это важно:
        в исходном наборе повреждённые значения лежат в `EFFECTIVE_YR`
        (640 записей с нечисловым мусором вида `1C`, `4A`, `/2`), а эта
        колонка не является признаком, поэтому проверка только признаков
        её бы пропустила.

        Returns:
            Для каждой колонки с нечисловыми значениями: их количество,
            доля и примеры.
        """
        result: dict[str, dict[str, Any]] = {}
        for column in (self.numeric_cols or []):
            if column not in df.columns:
                continue
            present = df[column].dropna()
            if present.empty:
                continue
            numeric = pd.to_numeric(present, errors="coerce")
            non_numeric = int(numeric.isna().sum())
            if non_numeric == 0:
                continue
            result[column] = {
                "non_numeric": non_numeric,
                "ratio": float(non_numeric / len(present)),
                "checked": int(len(present)),
                "examples": [str(value) for value in present[numeric.isna()].unique()[:5]],
            }
        return result

    # ------------------------------------------------------------------
    # Своевременность
    # ------------------------------------------------------------------

    def timeliness(
        self,
        df: pd.DataFrame,
        time_col: str = "INSR_BEGIN",
        date_format: str | None = None,
    ) -> dict[str, Any]:
        """Временные границы батча и максимальный разрыв между датами.

        Args:
            date_format: формат дат исходного набора. Используется как
                первый вариант разбора; если он не подходит (например, батч
                записан в ISO), разбор переходит к другим форматам.
        """
        if time_col not in df.columns:
            return {"available": False, "reason": f"колонка {time_col!r} отсутствует"}

        parsed = parse_datetime(df[time_col], date_format)
        dates = parsed.dropna().sort_values()
        if dates.empty:
            return {
                "available": False,
                "reason": f"не удалось разобрать даты в колонке {time_col!r}",
            }

        unique = dates.drop_duplicates()
        gaps = unique.diff().dropna()
        max_gap = int(gaps.max().days) if not gaps.empty else 0

        return {
            "available": True,
            "min_date": dates.min().isoformat(),
            "max_date": dates.max().isoformat(),
            "span_days": int((dates.max() - dates.min()).days),
            "n_unique_dates": int(len(unique)),
            "max_gap_days": max_gap,
            "unparsed_dates": int(parsed.isna().sum()),
        }

    # ------------------------------------------------------------------
    # Правила
    # ------------------------------------------------------------------

    def check_static_rules(self, df: pd.DataFrame) -> dict[str, float]:
        """Доля строк, нарушающих каждое статическое правило."""
        applicable = self.rules.available(df)
        ratios, _ = evaluate_rules(df, applicable)
        return ratios

    def skipped_rules(self, df: pd.DataFrame) -> list[str]:
        """Правила, которые невозможно проверить на этом батче."""
        return self.rules.missing(df)

    def check_dynamic_rules(
        self,
        df: pd.DataFrame,
        rules: list[dict[str, Any]],
        min_antecedent_rows: int = 30,
    ) -> dict[str, dict[str, Any]]:
        """Доля строк, нарушающих ассоциативные правила.

        Ключевое отличие от прежней реализации: доля нарушений считается
        относительно числа строк, удовлетворяющих антеценденту, а не всего
        батча. Иначе правило с поддержкой 1 % всегда давало бы долю
        нарушений порядка 0.01, а на практике — ровно 0.0, как и показал
        baseline-прогон.

        Args:
            min_antecedent_rows: минимальное число строк под антецендентом,
                при котором долю нарушений вообще имеет смысл считать.
                Без этого порога правило, совпавшее с одной-двумя строками
                малого батча, давало бы «100 % нарушений»: в прогоне
                наблюдались значения `violation_ratio = 1.0` при
                `antecedent_rows = 1`, что ничего не говорит о качестве
                данных. Такие случаи помечаются отдельно.

        Returns:
            Для каждого правила: `violation_ratio` (относительно антецендента),
            `support` (доля строк под антецендентом), абсолютные числа и
            признак достаточности выборки.
        """
        if not rules or df.empty:
            return {}

        # Категории приводим к строке один раз: значения в наборе смешанного типа.
        prepared = df.copy()
        for column in {item["column"] for rule in rules for item in rule["antecedents"]}:
            if column in prepared.columns:
                prepared[column] = prepared[column].astype("object").where(
                    prepared[column].notna(), other=pd.NA
                )

        result: dict[str, dict[str, Any]] = {}
        for rule in rules:
            antecedent = pd.Series(True, index=prepared.index)
            for item in rule["antecedents"]:
                if item["column"] not in prepared.columns:
                    antecedent = pd.Series(False, index=prepared.index)
                    break
                antecedent &= prepared[item["column"]].astype(str) == str(item["value"])

            antecedent_count = int(antecedent.sum())
            if antecedent_count == 0:
                result[rule["label"]] = {
                    "violation_ratio": None,
                    "support": 0.0,
                    "antecedent_rows": 0,
                    "violations": 0,
                    "sufficient_support": False,
                    "note": "антецендент не встречается в батче",
                }
                continue

            consequent = pd.Series(True, index=prepared.index)
            for item in rule["consequents"]:
                if item["column"] not in prepared.columns:
                    consequent = pd.Series(False, index=prepared.index)
                    break
                consequent &= prepared[item["column"]].astype(str) == str(item["value"])

            violations = int((antecedent & ~consequent).sum())
            sufficient = antecedent_count >= min_antecedent_rows
            result[rule["label"]] = {
                "violation_ratio": float(violations / antecedent_count) if sufficient else None,
                "raw_violation_ratio": float(violations / antecedent_count),
                "support": float(antecedent_count / len(prepared)),
                "antecedent_rows": antecedent_count,
                "violations": violations,
                "sufficient_support": sufficient,
                **({} if sufficient else {
                    "note": (
                        f"антецендент встречается лишь в {antecedent_count} строках "
                        f"(минимум {min_antecedent_rows}); доля нарушений не показана"
                    )
                }),
            }

        return result

    # ------------------------------------------------------------------
    # Очистка
    # ------------------------------------------------------------------

    def clean_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Очистить батч и вернуть отчёт о применённых правилах.

        Шаги:
        1. удаление строк, где доля пропусков выше порога;
        2. удаление дубликатов;
        3. удаление строк, нарушающих правила с `severity=error`, если доля
           нарушений по правилу не превышает порог.

        Итоги отчёта считаются от размера исходного батча, поэтому
        `rows_removed` действительно равно `rows_in - rows_out`, а сумма
        удалений по каждому правилу совпадает с итогом.
        """
        rows_in = int(len(df))
        report: dict[str, Any] = {
            "rows_in": rows_in,
            "dropped_by_missing": 0,
            "dropped_duplicates": 0,
            "max_row_missing_ratio_threshold": self.max_row_missing_ratio,
            "max_violation_ratio_threshold": self.max_violation_ratio,
        }

        result = df

        # 1. Пропуски
        if self.max_row_missing_ratio < 1.0 and not result.empty:
            columns = [
                column
                for column in (self.feature_cols or result.columns)
                if column in result.columns
            ]
            if columns:
                frame = result[columns]
                missing_ratio = frame.isna().sum(axis=1) / len(columns)
                keep = missing_ratio <= self.max_row_missing_ratio
                report["dropped_by_missing"] = int((~keep).sum())
                result = result.loc[keep]

        # 2. Дубликаты
        if not result.empty:
            before = len(result)
            result = result.drop_duplicates()
            report["dropped_duplicates"] = before - len(result)

        # 3. Правила качества
        unavailable = self.rules.missing(result)
        applicable = self.rules.available(result)
        result, rule_report = clean_by_rules(result, applicable, self.max_violation_ratio)

        # Правила, которые невозможно проверить (нет колонки), попадают
        # в тот же отчёт. Без этого они молча исчезали бы: `available()`
        # отфильтровывает их до вызова `clean_by_rules`, и тот уже не мог
        # их зафиксировать — а потеря контроля качества должна быть видна.
        for name in unavailable:
            rule = next(item for item in self.rules if item.name == name)
            rule_report["skipped"].append(
                {
                    "rule": name,
                    "condition": rule.describe(),
                    "violation_ratio": None,
                    "reason": "колонка отсутствует в батче — правило не проверялось",
                }
            )

        report.update(rule_report)

        rows_out = int(len(result))
        report["rows_out"] = rows_out
        report["rows_removed"] = rows_in - rows_out
        report["removed_ratio"] = (
            float(report["rows_removed"] / rows_in) if rows_in else 0.0
        )
        return result, report

    # ------------------------------------------------------------------
    # Сводный отчёт
    # ------------------------------------------------------------------

    def evaluate(
        self,
        df: pd.DataFrame,
        time_col: str,
        dynamic_rules: list[dict[str, Any]] | None = None,
        date_format: str | None = None,
    ) -> dict[str, Any]:
        """Полный набор метапараметров качества по батчу."""
        applicable = self.rules.available(df)
        static_ratios, static_counts = evaluate_rules(df, applicable)

        return {
            "rows": int(len(df)),
            "completeness": self.completeness(df),
            "validity": self.validity(df, applicable),
            "type_check": self.type_check(df),
            "timeliness": self.timeliness(df, time_col, date_format),
            "static_rule_violations": static_ratios,
            "static_rule_counts": static_counts,
            "dynamic_rule_violations": self.check_dynamic_rules(df, dynamic_rules or []),
            "rules_skipped": self.skipped_rules(df),
        }
