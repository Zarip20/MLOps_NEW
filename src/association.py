"""Ассоциативные правила (Apriori / FP-tree).

Переписан по результатам baseline-прогона (PLAN.md, §1.8). Исходная
реализация выдавала пять правил с `confidence = 1.0`, и все пять —
в сторону мажоритного класса `INSR_TYPE = 1202` (76 % данных). Причина
методическая: `nlargest(5, 'confidence')` при пороге 0.5 структурно
не может вернуть содержательное правило — максимальная уверенность
достигается именно на частом следствии.

Исправления:

* **D-4 — отбор по `lift`, а не по `confidence`.** Lift показывает, во сколько
  раз правило сильнее случайного совпадения, и потому не «выбирает»
  мажоритный класс. Дополнительно следствия, совпадающие с самым частым
  значением своей колонки, отбрасываются как тривиальные.
* **D-16 — правила хранятся структурно** (`{"column": ..., "value": ...}`),
  а не разбираются из строк вида `"MAKE = AFRO"`. Прежний разбор ломался
  на любом значении, содержащем `=`, и требовал замены подстрок.
* Пороги (`min_support`, `min_confidence`, `min_lift`, `max_len`, `n_rules`)
  и алгоритм перенесены в конфигурацию; по умолчанию `fpgrowth` — задание
  прямо допускает «Apriori / FP-tree», а FP-tree в mlxtend в 3–6 раз быстрее.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

ALGORITHMS = ("fpgrowth", "apriori")


def _to_transaction_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Представить категориальные признаки в виде набора «признак = значение»."""
    present = [column for column in columns if column in df.columns]
    if not present:
        return pd.DataFrame(index=df.index)

    frame = df[present].copy()
    for column in present:
        # Пропуски получают явную метку, чтобы не терять транзакцию.
        frame[column] = frame[column].astype(object).where(
            frame[column].notna(), other="__MISSING__"
        ).astype(str)

    return pd.get_dummies(frame, prefix_sep="=", dtype=bool)


def _majority_values(frame: pd.DataFrame) -> dict[str, str]:
    """Самое частое значение каждой колонки — кандидат на тривиальное следствие."""
    majority: dict[str, str] = {}
    for column in frame.columns:
        counts = frame[column].value_counts()
        if not counts.empty:
            majority[column] = str(counts.index[0])
    return majority


def _item_to_pair(item: str) -> dict[str, str]:
    """Разобрать элемент набора вида `MAKE=AFRO` в пару колонка/значение.

    Разделяем по первому знаку `=`: значение категории может содержать
    `=` (или начинаться с него), и прежняя замена всех вхождений ломала
    такие случаи.
    """
    if "=" not in item:
        return {"column": item, "value": ""}
    column, value = item.split("=", 1)
    return {"column": column, "value": value}


def _label(antecedents: list[dict[str, str]], consequents: list[dict[str, str]]) -> str:
    """Человекочитаемое имя правила для метаданных и отчёта."""
    left = " ∧ ".join(f"{item['column']} = {item['value']}" for item in antecedents)
    right = " ∨ ".join(f"{item['column']} = {item['value']}" for item in consequents)
    return f"{left} → {right}"


def generate_association_rules(
    df: pd.DataFrame,
    categorical_cols: Sequence[str],
    output_path: str | Path,
    *,
    min_support: float = 0.01,
    min_confidence: float = 0.5,
    min_lift: float = 1.2,
    max_len: int | None = 2,
    n_rules: int = 5,
    algorithm: str = "fpgrowth",
    exclude_majority_consequent: bool = True,
) -> list[dict[str, Any]]:
    """Найти ассоциативные правила и сохранить их.

    Args:
        df: батч данных.
        categorical_cols: категориальные признаки для анализа.
        output_path: куда сохранить правила в формате JSON.
        min_support: минимальная поддержка набора.
        min_confidence: минимальная уверенность правила.
        min_lift: минимальный lift — отсекает правила, не сильнее случайных.
        max_len: максимальная длина набора; ограничивает комбинаторный рост.
        n_rules: сколько правил сохранить.
        algorithm: `fpgrowth` или `apriori`.
        exclude_majority_consequent: не сохранять правила, следствие которых
            совпадает с самым частым значением своей колонки.

    Returns:
        Список правил; каждый элемент пригоден для `json.dump` без преобразований.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Алгоритм должен быть одним из {ALGORITHMS}, получено {algorithm!r}")

    transactions = _to_transaction_frame(df, categorical_cols)
    if transactions.empty or transactions.shape[1] < 2:
        return _persist([], output_path)

    # У apriori и fpgrowth в mlxtend одинаковый набор параметров.
    finder = fpgrowth if algorithm == "fpgrowth" else apriori
    itemsets = finder(
        transactions,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )

    if itemsets.empty:
        return _persist([], output_path)

    candidates = association_rules(
        itemsets, metric="confidence", min_threshold=min_confidence
    )
    if candidates.empty:
        return _persist([], output_path)

    majority = _majority_values(transactions) if exclude_majority_consequent else {}
    filtered = candidates.loc[
        candidates["consequents"].apply(lambda itemset: not _is_majority(itemset, majority))
    ]
    if filtered.empty:
        return _persist([], output_path)

    filtered = filtered[filtered["lift"] >= min_lift]
    if filtered.empty:
        return _persist([], output_path)

    # Ранжирование по lift, при равенстве — по уверенности и поддержке.
    ranked = filtered.sort_values(
        by=["lift", "confidence", "support"], ascending=False
    ).head(n_rules)

    rules: list[dict[str, Any]] = []
    for _, row in ranked.iterrows():
        antecedents = [_item_to_pair(item) for item in row["antecedents"]]
        consequents = [_item_to_pair(item) for item in row["consequents"]]
        rules.append(
            {
                "label": _label(antecedents, consequents),
                "antecedents": antecedents,
                "consequents": consequents,
                "support": float(row["support"]),
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"]),
            }
        )

    return _persist(rules, output_path)


def _is_majority(itemset: frozenset, majority: dict[str, str]) -> bool:
    """Следствие целиком совпадает с мажоритным значением своей колонки."""
    for item in itemset:
        pair = _item_to_pair(item)
        if majority.get(pair["column"]) == pair["value"]:
            return True
    return False


def _persist(rules: list[dict[str, Any]], output_path: str | Path) -> list[dict[str, Any]]:
    """Записать правила в JSON. Отсутствие правил — не ошибка."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rules, handle, ensure_ascii=False, indent=2)
    return rules


def load_association_rules(rules_path: str | Path) -> list[dict[str, Any]]:
    """Прочитать правила; отсутствие файла — пустой список, не ошибка."""
    path = Path(rules_path)
    if not path.is_file():
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def rule_statistics(rules: list[dict[str, Any]]) -> dict[str, Any]:
    """Сводка по найденным правилам для метаданных."""
    if not rules:
        return {"n_rules": 0}
    lifts = [rule["lift"] for rule in rules]
    return {
        "n_rules": len(rules),
        "max_lift": round(max(lifts), 4),
        "mean_lift": round(sum(lifts) / len(lifts), 4),
        "mean_confidence": round(sum(rule["confidence"] for rule in rules) / len(rules), 4),
        "mean_support": round(sum(rule["support"] for rule in rules) / len(rules), 6),
    }
