"""Декларативные правила качества данных (АР-14).

Заменяет прежнюю схему, где условие правила задавалось строкой
`"lambda df: ..."` и выполнялось через `eval()`. Теперь правило —
это данные, которые можно проверить, показать в отчёте и отличить
друг от друга по намерению.

Главное, что даёт разделение `on_missing`: пропуск в `PREMIUM` — это
неполнота, а пропуск `CARRYING_CAPACITY` у грузовика — настоящее
нарушение. Старый код (сравнение `NaN >= 0` → `False`) смешивал эти
вещи и из-за этого удалял пропуски, объявляя их нарушениями (D-27).

Правило описывает условие, которому строка **должна удовлетворять**.
Если строка условию не удовлетворяет — это нарушение, и всё.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Condition:
    """Простое условие над одной колонкой."""

    column: str
    op: str
    value: Any = None

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return evaluate_condition(self.column, self.op, self.value, df)


@dataclass(frozen=True)
class Rule:
    """Правило качества данных.

    Attributes:
        name: идентификатор правила, попадает в метаданные и отчёт.
        column: проверяемая колонка.
        op: оператор из списка `config.VALID_OPERATORS`.
        value: операнд (скаляр, список или пара границ).
        applies_when: необязательное условие применимости. Если задано,
            проверяются только те строки, для которых оно истинно.
        on_missing: как трактовать пропуск в проверяемой колонке —
            `ignore` (пропуск не нарушение) или `violation`
            (пропуск считается нарушением).
        severity: `error` — влияет на очистку; `warn` — только мониторинг.
        description: человекочитаемое пояснение для отчёта.
    """

    name: str
    column: str
    op: str
    value: Any = None
    applies_when: Condition | None = None
    on_missing: str = "ignore"
    severity: str = "error"
    description: str = ""

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def applicable_mask(self, df: pd.DataFrame) -> pd.Series:
        """Маска строк, к которым правило вообще применимо."""
        if self.applies_when is None:
            return pd.Series(True, index=df.index)
        return self.applies_when.evaluate(df).fillna(False).astype(bool)

    def holds_mask(self, df: pd.DataFrame) -> pd.Series:
        """Маска строк, удовлетворяющих условию правила."""
        return evaluate_condition(self.column, self.op, self.value, df)

    def violation_mask(self, df: pd.DataFrame) -> pd.Series:
        """Маска строк-нарушений с учётом `applies_when` и `on_missing`."""
        holds = self.holds_mask(df).fillna(False).astype(bool)
        applicable = self.applicable_mask(df)

        # Операторы, предметом которых является сам пропуск, не требуют
        # отдельной обработки отсутствующих значений.
        if self.op in ("notna", "isna"):
            violated = ~holds
        elif self.on_missing == "ignore":
            missing = df[self.column].isna()
            violated = ~holds & ~missing
        else:
            violated = ~holds

        return (violated & applicable).fillna(False).astype(bool)

    def describe(self) -> str:
        """Человекочитаемая формулировка условия для отчёта."""
        if self.description:
            return self.description
        condition = f"{self.column} {_operator_text(self.op)} {_value_text(self.value)}"
        if self.applies_when is not None:
            scope = f"{self.applies_when.column} {_operator_text(self.applies_when.op)} " \
                    f"{_value_text(self.applies_when.value)}"
            return f"{condition}, если {scope}"
        return condition


@dataclass
class RuleSet:
    """Набор правил, построенный из конфигурации."""

    rules: list[Rule] = field(default_factory=list)

    @classmethod
    def from_config(cls, raw_rules: Sequence[dict[str, Any]]) -> "RuleSet":
        rules: list[Rule] = []
        for spec in raw_rules:
            applies_when = None
            if spec.get("applies_when"):
                applies_when = Condition(
                    column=spec["applies_when"]["column"],
                    op=spec["applies_when"]["op"],
                    value=spec["applies_when"].get("value"),
                )
            rules.append(
                Rule(
                    name=spec["name"],
                    column=spec["column"],
                    op=spec["op"],
                    value=spec.get("value"),
                    applies_when=applies_when,
                    on_missing=spec.get("on_missing", "ignore"),
                    severity=spec.get("severity", "error"),
                    description=spec.get("description", ""),
                )
            )
        return cls(rules=rules)

    def __iter__(self):
        return iter(self.rules)

    def __len__(self) -> int:
        return len(self.rules)

    def error_rules(self) -> list[Rule]:
        return [rule for rule in self.rules if rule.is_error]

    def warn_rules(self) -> list[Rule]:
        return [rule for rule in self.rules if not rule.is_error]

    def available(self, df: pd.DataFrame) -> "RuleSet":
        """Отбросить правила, чьи колонки отсутствуют в датафрейме.

        Батч может прийти с неполной схемой; такое правило не должно
        ронять конвейер, но должно быть видно в метаданных как
        «не проверялось».
        """
        kept = [rule for rule in self.rules if _rule_columns_present(rule, df)]
        return RuleSet(rules=kept)

    def missing(self, df: pd.DataFrame) -> list[str]:
        """Имена правил, которые невозможно проверить на этом батче."""
        return [
            rule.name
            for rule in self.rules
            if not _rule_columns_present(rule, df)
        ]


def _rule_columns_present(rule: Rule, df: pd.DataFrame) -> bool:
    if rule.column not in df.columns:
        return False
    if rule.applies_when is not None and rule.applies_when.column not in df.columns:
        return False
    return True


def evaluate_condition(
    column: str,
    op: str,
    value: Any,
    df: pd.DataFrame,
) -> pd.Series:
    """Вычислить булеву маску условия по колонке.

    Все сравнения оборачиваются в `try/except`: неизвестная колонка или
    несовместимые типы не должны прерывать конвейер — вызывающий код
    обрабатывает такие правила через `RuleSet.available()`.
    """
    if column not in df.columns:
        raise KeyError(f"Колонка {column!r} отсутствует в датафрейме")

    series = df[column]

    if op == "notna":
        return series.notna()
    if op == "isna":
        return series.isna()
    if op == "positive":
        return _numeric(series) > 0
    if op == "non_negative":
        return _numeric(series) >= 0
    if op == "eq":
        return _compare(series, value, lambda left, right: left == right)
    if op == "ne":
        return _compare(series, value, lambda left, right: left != right)
    if op == "ge":
        return _compare(series, value, lambda left, right: left >= right)
    if op == "gt":
        return _compare(series, value, lambda left, right: left > right)
    if op == "le":
        return _compare(series, value, lambda left, right: left <= right)
    if op == "lt":
        return _compare(series, value, lambda left, right: left < right)
    if op == "in":
        return series.isin(list(value))
    if op == "not_in":
        return ~series.isin(list(value))
    if op == "between":
        low, high = value
        numeric = _numeric(series)
        return (numeric >= low) & (numeric <= high)
    if op == "not_between":
        low, high = value
        numeric = _numeric(series)
        return ~((numeric >= low) & (numeric <= high))

    raise ValueError(f"Неизвестный оператор правила: {op!r}")


def _numeric(series: pd.Series) -> pd.Series:
    """Числовое представление колонки; нечисловые значения дают NaN.

    Это позволяет корректно применять числовые операторы к колонкам со
    смешанным типом (см. D-26: в `EFFECTIVE_YR` встречается нечисловой мусор).
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series, errors="coerce")


def _compare(series: pd.Series, value: Any, operation) -> pd.Series:
    """Сравнение с приведением к числу, если это уместно."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return operation(_numeric(series), value)
    return operation(series, value)


def _operator_text(op: str) -> str:
    return {
        "eq": "==", "ne": "!=", "ge": ">=", "gt": ">", "le": "<=", "lt": "<",
        "between": "в диапазоне", "not_between": "вне диапазона",
        "in": "в списке", "not_in": "не в списке",
        "notna": "заполнено", "isna": "пусто",
        "positive": "> 0", "non_negative": ">= 0",
    }.get(op, op)


def _value_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


# ---------------------------------------------------------------------------
# Применение правил
# ---------------------------------------------------------------------------


def evaluate_rules(
    df: pd.DataFrame,
    rules: Iterable[Rule],
) -> tuple[dict[str, float], dict[str, int]]:
    """Посчитать долю нарушений по каждому правилу.

    Returns:
        (доли нарушений, число нарушенных строк) — обе структуры
        индексированы именами правил.
    """
    ratios: dict[str, float] = {}
    counts: dict[str, int] = {}
    total = max(len(df), 1)

    for rule in rules:
        violations = rule.violation_mask(df)
        count = int(violations.sum())
        counts[rule.name] = count
        ratios[rule.name] = count / total

    return ratios, counts


def clean_by_rules(
    df: pd.DataFrame,
    rules: Iterable[Rule],
    max_violation_ratio: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Удалить строки, нарушающие правила с severity=error.

    Правило применяется, только если доля нарушений по нему не превышает
    `max_violation_ratio`. Иначе считается, что правило настроено неверно
    для этих данных: массовое удаление уничтожило бы датасет, поэтому такое
    правило помечается как неприменённое и попадает в отчёт.

    Returns:
        (очищенный датафрейм, отчёт о применении правил). Отчёт содержит
        только сведения о правилах; итоговые `rows_in`/`rows_out` считает
        вызывающий код, чтобы не было двух конкурирующих точек отсчёта.
    """
    report: dict[str, Any] = {
        "applied": [],
        "skipped": [],
        "dropped_by_rule": {},
    }

    result = df
    for rule in rules:
        if not rule.is_error:
            continue
        if not _rule_columns_present(rule, result):
            report["skipped"].append({**rule.describe(), "reason": "колонка отсутствует"})
            continue
        if result.empty:
            break

        violations = rule.violation_mask(result)
        ratio = float(violations.mean())

        if ratio <= max_violation_ratio:
            before = len(result)
            result = result.loc[~violations]
            dropped = before - len(result)
            report["dropped_by_rule"][rule.name] = dropped
            report["applied"].append(
                {
                    "rule": rule.name,
                    "condition": rule.describe(),
                    "violation_ratio": round(ratio, 6),
                    "dropped_rows": dropped,
                }
            )
        else:
            report["skipped"].append(
                {
                    "rule": rule.name,
                    "condition": rule.describe(),
                    "violation_ratio": round(ratio, 6),
                    "reason": (
                        f"доля нарушений {ratio:.4f} превышает порог "
                        f"{max_violation_ratio}; правило не применено"
                    ),
                }
            )

    report["rows_removed_by_rules"] = sum(report["dropped_by_rule"].values())
    return result, report
