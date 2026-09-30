"""Текстовый отчёт о работе системы.

Требование §4 задания 1: отчёт должен отражать изменение качества данных,
метрик лучшей модели, **отобранных гиперпараметров** и **производительности
модели**. Исходная реализация писала только полноту и f1 с roc_auc —
гиперпараметров и времени в отчёте не было вовсе.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from src.config import Config
from src.state import StateStore
from src.utils import format_duration, load_json, read_artifacts, save_json

logger = logging.getLogger(__name__)


def _read_metadata(config: Config, prefix: str) -> list[dict[str, Any]]:
    """Прочитать все файлы метаданных с заданным префиксом, по индексу батча."""
    return read_artifacts(config.metadata_dir, prefix)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "н/д"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(rows: list[list[str]], headers: list[str]) -> list[str]:
    """Собрать текстовую таблицу с выравниванием по ширине колонок."""
    if not rows:
        return []
    widths = [
        max(len(str(headers[index])), *(len(str(row[index])) for row in rows))
        for index in range(len(headers))
    ]
    line = "  ".join("-" * width for width in widths)
    output = ["  ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)), line]
    output.extend(
        "  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row))
        for row in rows
    )
    return output


def build_report(config: Config, state: StateStore) -> Path:
    """Построить отчёт и вернуть путь к нему."""
    quality = _read_metadata(config, "quality")
    metrics = _read_metadata(config, "model_metrics")
    manifests = _read_metadata(config, "run_manifest")

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.reports_dir / f"summary_{stamp}.txt"

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("ОТЧЁТ О РАБОТЕ ML-СИСТЕМЫ ОБРАБОТКИ ПОТОКОВЫХ ДАННЫХ")
    lines.append("=" * 78)
    lines.append(f"Сформирован:       {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Хэш конфигурации:  {config.config_hash}")
    lines.append(f"Батчей всего:      {len(state)}")
    lines.append(f"Батчей обработано: {state.last_processed + 1}")
    lines.append("")

    lines.extend(_section_run(manifests, state))
    lines.extend(_section_quality(quality))
    lines.extend(_section_rules(config, quality))
    lines.extend(_section_metrics(metrics))
    lines.extend(_section_best(metrics, manifests))
    lines.append("")
    lines.append("  6. ДРЕЙФ ДАННЫХ")
    lines.append("  " + "-" * 74)
    lines.extend(_drift_lines(manifests))
    lines.append("")
    lines.append("  7. РЕЕСТР МОДЕЛЕЙ И ГЕЙТ КАЧЕСТВА")
    lines.append("  " + "-" * 74)
    lines.extend(_registry_lines(manifests))
    lines.append("")

    lines.extend(_section_hyperparameters(config))
    lines.extend(_section_performance(manifests))
    lines.extend(_section_meta(load_json(config.reports_dir / "meta.json")))
    lines.extend(_section_sweep(config))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Машиночитаемая копия — основа для дашборда и для Meta Learning.
    save_json(
        config.reports_dir / f"summary_{stamp}.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "config_hash": config.config_hash,
            "batches_total": len(state),
            "batches_processed": state.last_processed + 1,
            "quality": quality,
            "metrics": metrics,
            "manifests": manifests,
            "meta": load_json(config.reports_dir / "meta.json"),
        },
    )

    logger.info("Отчёт сохранён: %s", path.name)
    return path


def _section_run(manifests: list[dict[str, Any]], state: StateStore) -> list[str]:
    """Состояние прогона и сведения об окружении."""
    lines = ["-" * 78, "1. СОСТОЯНИЕ ПРОГОНА", "-" * 78]
    if not manifests:
        lines.append("Нет выполненных прогонов.")
        lines.append("")
        return lines

    environment = manifests[-1].get("environment", {})
    for key in ("python", "numpy", "pandas", "scikit_learn"):
        if key in environment:
            lines.append(f"  {key:<18} {environment[key]}")

    store = manifests[-1].get("training_store", {})
    if store.get("enabled"):
        lines.append(
            f"  {'накоплено':<18} {store.get('rows', 0)} строк "
            f"из {store.get('parts', 0)} батчей"
        )

    total = sum(
        float(item.get("performance", {}).get("durations", {}).get("train", 0.0))
        for item in manifests
    )
    if total:
        lines.append(f"  {'время обучения':<18} {format_duration(total)} суммарно")
    lines.append("")
    return lines


def _section_quality(quality: list[dict[str, Any]]) -> list[str]:
    """Полнота, валидность, типы и очистка по батчам."""
    lines = ["-" * 78, "2. КАЧЕСТВО ДАННЫХ ПО БАТЧАМ", "-" * 78]
    if not quality:
        lines.append("Нет данных о качестве.")
        lines.append("")
        return lines

    rows: list[list[str]] = []
    for item in quality:
        index = item.get("batch_idx", "?")
        completeness = item.get("completeness", {})
        cleaning = item.get("cleaning", {})
        target = item.get("target", {})
        rows.append(
            [
                str(index),
                _fmt(completeness.get("total_missing_ratio")),
                _fmt(completeness.get("max_col_missing_ratio")),
                _fmt(completeness.get("max_row_missing_ratio")),
                str(cleaning.get("rows_removed", 0)),
                _fmt(target.get("positive_rate")),
            ]
        )
    lines.extend(
        _table(
            rows,
            ["батч", "полнота", "худшая\nколонка", "худшая\nстрока", "удалено", "доля POS"],
        )
    )
    lines.append("")
    lines.append("  (Полнота рассчитана по колонкам признаков, без целевой метки.)")
    lines.append("")

    # Применённые правила очистки — суммарно по всем батчам.
    applied: dict[str, dict[str, Any]] = {}
    for item in quality:
        for entry in item.get("cleaning", {}).get("applied", []):
            record = applied.setdefault(
                entry["rule"],
                {"condition": entry["condition"], "rows": 0, "batches": 0},
            )
            record["rows"] += entry["dropped_rows"]
            record["batches"] += 1

    lines.append("  Применённые правила очистки:")
    if applied:
        rule_rows = [
            [name, f"{data['batches']}", str(data["rows"]), data["condition"]]
            for name, data in sorted(applied.items(), key=lambda pair: -pair[1]["rows"])
        ]
        lines.extend(_table(rule_rows, ["правило", "батчей", "строк удалено", "условие"]))
    else:
        lines.append("    ни одно правило не привело к удалению строк")
    lines.append("")

    # Проблемы формата, найденные проверкой типов.
    type_issues: dict[str, int] = {}
    for item in quality:
        for column, info in item.get("type_check", {}).items():
            type_issues[column] = type_issues.get(column, 0) + int(info.get("non_numeric", 0))
    if type_issues:
        lines.append("  Нечисловые значения в числовых колонках:")
        for column, count in sorted(type_issues.items(), key=lambda pair: -pair[1]):
            lines.append(f"    {column:<24} {count}")
        lines.append("")
    return lines


def _section_rules(config: Config, quality: list[dict[str, Any]]) -> list[str]:
    """Ассоциативные правила и динамические нарушения."""
    from src.association import load_association_rules

    lines = ["-" * 78, "3. АССОЦИАТИВНЫЕ ПРАВИЛА И КОНТРОЛЬ КОРРЕКТНОСТИ", "-" * 78]
    rules = load_association_rules(config.rules_file)
    if not rules:
        lines.append("  Правила не найдены.")
        lines.append("")
        return lines

    lines.append(
        f"  Найдено правил: {len(rules)} "
        f"(отбор по lift ≥ {config.association.get('min_lift')}, "
        f"следствие ≠ мажоритный класс)"
    )
    lines.append("")
    rule_rows = [
        [
            str(index + 1),
            " ∧ ".join(f"{i['column']}={i['value']}" for i in rule["antecedents"]),
            " ∨ ".join(f"{i['column']}={i['value']}" for i in rule["consequents"]),
            _fmt(rule["support"], 4),
            _fmt(rule["confidence"], 4),
            _fmt(rule["lift"], 3),
        ]
        for index, rule in enumerate(rules)
    ]
    lines.extend(_table(rule_rows, ["№", "антецендент", "следствие", "support", "confid", "lift"]))
    lines.append("")

    # Нарушения, накопленные по батчам.
    totals: dict[str, dict[str, Any]] = {}
    for item in quality:
        for label, entry in (item.get("dynamic_rule_violations") or {}).items():
            record = totals.setdefault(
                label,
                {"violations": 0, "antecedent": 0, "batches": 0, "insufficient": 0, "ratios": []},
            )
            record["violations"] += int(entry.get("violations", 0))
            record["antecedent"] += int(entry.get("antecedent_rows", 0))
            record["batches"] += 1
            if not entry.get("sufficient_support", True):
                record["insufficient"] += 1
            elif entry.get("violation_ratio") is not None:
                record["ratios"].append(entry["violation_ratio"])

    if not totals:
        lines.append("  Динамические нарушения не рассчитывались.")
        lines.append("")
        return lines

    lines.append("  Доля строк, нарушающих правило (относительно антецендента):")
    violation_rows = []
    for label, record in sorted(
        totals.items(),
        key=lambda pair: -(sum(pair[1]["ratios"]) / len(pair[1]["ratios"]) if pair[1]["ratios"] else 0.0),
    ):
        mean_ratio = sum(record["ratios"]) / len(record["ratios"]) if record["ratios"] else None
        note = ""
        if record["insufficient"]:
            note = f"мало антецендента в {record['insufficient']} батчах"
        violation_rows.append(
            [
                label,
                _fmt(mean_ratio),
                str(record["violations"]),
                str(record["antecedent"]),
                str(record["batches"]),
                note,
            ]
        )
    lines.extend(
        _table(
            violation_rows,
            ["правило", "средняя\nдоля", "нарушений", "строк под\nантецендентом", "батчей", ""],
        )
    )
    lines.append("")
    lines.append(
        f"  Порог достаточной выборки под антецендентом: "
        f"{config.association.get('min_antecedent_rows')} строк."
    )
    lines.append("")
    return lines


def _section_metrics(metrics: list[dict[str, Any]]) -> list[str]:
    """Метрики моделей по батчам."""
    lines = ["-" * 78, "4. МЕТРИКИ МОДЕЛЕЙ ПО БАТЧАМ", "-" * 78]
    if not metrics:
        lines.append("Нет данных о моделях.")
        lines.append("")
        return lines

    names: list[str] = []
    for item in metrics:
        for name in item:
            if name not in ("batch_idx", "batch") and name not in names:
                names.append(name)

    headers = ["батч"] + [
        f"{name}\n{field}" for name in names for field in ("f1", "roc_auc", "recall")
    ]
    rows: list[list[str]] = []
    for item in metrics:
        row = [str(item.get("batch_idx", "?"))]
        for name in names:
            entry = item.get(name, {})
            row.extend(
                [
                    _fmt(entry.get("f1")),
                    _fmt(entry.get("roc_auc")),
                    _fmt(entry.get("recall")),
                ]
            )
        rows.append(row)
    lines.extend(_table(rows, headers))
    lines.append("")
    lines.append("  (Метрики рассчитаны на отложенной валидационной части батча, 20 %.)")
    lines.append("")
    return lines


def _section_best(
    metrics: list[dict[str, Any]], manifests: list[dict[str, Any]]
) -> list[str]:
    """Лучшая модель и её динамика."""
    lines = ["-" * 78, "5. ЛУЧШАЯ МОДЕЛЬ", "-" * 78]
    if not metrics:
        lines.append("Нет данных о моделях.")
        lines.append("")
        return lines

    last = metrics[-1]
    candidates = {
        name: entry
        for name, entry in last.items()
        if name not in ("batch_idx", "batch") and isinstance(entry, dict)
    }
    if not candidates:
        lines.append("Нет обученных моделей.")
        lines.append("")
        return lines

    best = max(candidates, key=lambda name: candidates[name].get("f1", 0.0))
    entry = candidates[best]
    lines.append(f"  Лучшая модель в последнем батче ({last.get('batch_idx')}): {best}")
    lines.append(f"    f1        = {_fmt(entry.get('f1'))}")
    lines.append(f"    precision = {_fmt(entry.get('precision'))}")
    lines.append(f"    recall    = {_fmt(entry.get('recall'))}")
    lines.append(f"    roc_auc   = {_fmt(entry.get('roc_auc'))}")
    lines.append(f"    accuracy  = {_fmt(entry.get('accuracy'))}")

    confusion = entry.get("confusion", {})
    if confusion:
        lines.append(
            f"    матрица ошибок: TN={confusion.get('tn')} FP={confusion.get('fp')} "
            f"FN={confusion.get('fn')} TP={confusion.get('tp')}"
        )
    lines.append("")

    # Динамика f1 по батчам — признак дрейфа модели.
    series: list[tuple[int, float]] = []
    for item in metrics:
        for name, value in item.items():
            if name == best and isinstance(value, dict):
                series.append((int(item["batch_idx"]), float(value.get("f1", 0.0))))
    if len(series) > 1:
        first, last_value = series[0][1], series[-1][1]
        change = last_value - first
        arrow = "снизилась" if change < 0 else "выросла"
        lines.append(
            f"  Динамика f1 модели {best}: {first:.4f} → {last_value:.4f} "
            f"({arrow} на {abs(change):.4f} за {len(series)} батчей)"
        )

    # Доля положительного класса — источник дрейфа целевой метки.
    rates: list[tuple[int, float]] = []
    for item in manifests:
        target = item.get("data", {}).get("target") or {}
        if target.get("positive_rate") is not None:
            rates.append((int(item["batch_idx"]), float(target["positive_rate"])))
    if len(rates) > 1:
        lines.append(
            f"  Доля положительного класса: {rates[0][1]:.4f} → {rates[-1][1]:.4f} "
            f"(изменение {(rates[-1][1] / rates[0][1] - 1) * 100:+.1f} %)"
        )
    lines.append("")
    return lines


def _drift_lines(manifests: list[dict[str, Any]]) -> list[str]:
    """Дрейф: статусы по батчам, PSI/KS и динамика доли положительного класса."""
    entries = [
        (item["batch_idx"], item.get("drift") or {})
        for item in manifests
        if item.get("drift")
    ]
    if not entries:
        return ["  Мониторинг дрейфа не выполнялся."]

    statuses: dict[str, int] = {}
    for _, drift in entries:
        name = str(drift.get("status", "unknown"))
        statuses[name] = statuses.get(name, 0) + 1

    # Пороги берутся из первого батча, где дрейф уже рассчитывался: на
    # батче-эталоне их ещё нет, и в отчёт попадало бы "None".
    thresholds = {}
    for _, drift in entries:
        if drift.get("thresholds"):
            thresholds = drift["thresholds"]
            break
    lines = [
        "  Статусы: "
        + ", ".join(f"{name} — {count}" for name, count in sorted(statuses.items())),
        f"  Пороги: PSI warn {thresholds.get('psi_warn')}, "
        f"alert {thresholds.get('psi_alert')}, KS warn {thresholds.get('ks_warn')}",
        "",
    ]

    rows = []
    for index, drift in entries:
        target = drift.get("target") or {}
        change = target.get("relative_change")
        rows.append([
            str(index),
            str(drift.get("status", "-")),
            _fmt(drift.get("max_psi")),
            _fmt(drift.get("max_ks")),
            _fmt(target.get("psi")),
            f"{change:+.1f} %" if change is not None else "-",
        ])
    lines.extend(_table(rows, ["батч", "статус", "max PSI", "max KS", "PSI метки", "изм. POS"]))
    lines.append("")
    lines.append(
        "  Дрейф не прерывает конвейер: он фиксируется в метаданных и учитывается\n"
        "  гейтом качества при выборе продукта. Пороги подобраны по фактическому\n"
        "  распределению PSI: в наборе выражена сезонность, поэтому «учебные»\n"
        "  0.10/0.25 помечали бы аномалией большинство батчей."
    )
    return lines


def _registry_lines(manifests: list[dict[str, Any]]) -> list[str]:
    """Реестр версий: статусы, продуктовая версия, версии, прошедшие гейт."""
    if not manifests:
        return ["  Реестр пуст."]

    registry = manifests[-1].get("registry") or {}
    versions = registry.get("versions") or []
    if not versions:
        return ["  Реестр пуст."]

    statuses: dict[str, int] = {}
    for entry in versions:
        name = str(entry.get("status", "-"))
        statuses[name] = statuses.get(name, 0) + 1

    # Очистка старых файлов не должна выглядеть как потеря истории:
    # в реестре остаются все записи, поэтому здесь это указано явно.
    # Числа берутся из счётчиков реестра, а не из списка версий: в
    # манифесте записи урезаны до ключевых полей, там нет ни признака
    # очистки, ни полного перечня. Смешивать эти источники нельзя —
    # из урезанного списка получилось бы отрицательное число файлов.
    pruned = int(registry.get("n_artifacts_pruned") or 0)
    total = int(registry.get("n_versions") or len(versions))

    lines = [
        f"  Всего версий: {total}",
        "  Статусы: " + ", ".join(f"{k} — {v}" for k, v in sorted(statuses.items())),
    ]
    if pruned and pruned <= total:
        lines.append(
            f"  Файлы версий: {total - pruned} хранятся, "
            f"{pruned} удалены как устаревшие "
            f"(метрики всех версий сохранены в реестре)"
        )
    elif pruned:
        # Счётчики разошлись — это дефект данных, а не очистка.
        # Молчаливое «0 хранится» было бы хуже явного признака.
        lines.append(
            f"  ВНИМАНИЕ: счётчик удалённых файлов ({pruned}) больше "
            f"общего числа версий ({total}) — реестр требует проверки"
        )

    production = registry.get("production")
    if production:
        lines.append(
            f"  Продуктовая: {production.get('model')} {production.get('version')} "
            f"(батч {production.get('batch_idx')}, f1 = {_fmt(production.get('f1'))})"
        )
    lines.append("")

    promoted = [e for e in versions if e.get("status") in ("production", "previous")]
    if promoted:
        lines.append("  Версии, прошедшие гейт:")
        lines.extend(_table(
            [
                [str(e.get("version", "")), str(e.get("model", "")),
                 str(e.get("batch_idx", "")), _fmt(e.get("f1"))]
                for e in promoted
            ],
            ["версия", "модель", "батч", "f1"],
        ))
        lines.append("")

    rejection = registry.get("last_rejection")
    if rejection:
        lines.append("  Последний отказ:")
        for reason in rejection.get("reasons", []):
            lines.append(f"    - {reason}")
        lines.append("")

    lines.append("  Отказ не удаляет версию из истории: без этого нельзя понять, почему упало качество.")
    return lines


def _section_hyperparameters(config: Config) -> list[str]:
    """Отобранные гиперпараметры — требование §4 задания 1."""
    lines = ["-" * 78, "8. ОТОБРАННЫЕ ГИПЕРПАРАМЕТРЫ", "-" * 78]
    lines.append("")

    def dump(title: str, payload: dict[str, Any], indent: str = "    ") -> None:
        lines.append(f"  {title}:")
        for key, value in payload.items():
            lines.append(f"{indent}{key:<26} {value}")

    dump("mlp (нейронная сеть)", config.models.get("mlp", {}))
    lines.append("")
    dump("dt (дерево решений)", config.models.get("dt", {}))
    lines.append("")
    dump("валидация", config.validation)
    lines.append("")
    dump("предобработка", config.data.get("preprocessing", {}))
    lines.append("")
    dump("ассоциативные правила", config.association)
    lines.append("")
    return lines


def _section_meta(payload: dict[str, Any] | None) -> list[str]:
    """Meta Learning: что в прогоне влияло на качество (7.b.iii)."""
    lines = ["-" * 78, "10. META LEARNING: ВЛИЯНИЕ НА КАЧЕСТВО", "-" * 78]
    lines.append("")

    if not payload or payload.get("verdict") != "ok":
        note = (payload or {}).get("note", "анализ не выполнялся")
        lines.append(f"  {note}")
        lines.append("")
        return lines

    metric = payload.get("metric", "f1")
    lines.append(
        f"  Прогонов проанализировано: {payload.get('n_runs', 0)} "
        f"(батчей {payload.get('n_batches', 0)}), метрика {metric}"
    )
    lines.append("")

    findings = payload.get("findings") or []
    if findings:
        lines.append("  Выводы:")
        for item in findings:
            lines.append(f"    · {item}")
        lines.append("")

    varied = (payload.get("settings_influence") or {}).get("varied") or []
    lines.append("  Настройки, которые менялись внутри одной модели:")
    if varied:
        lines.append(
            "    (сравнение только внутри семейства: у каждой модели свои"
        )
        lines.append(
            "     значения параметров, и сравнение между моделями измеряло бы"
        )
        lines.append("     разницу моделей, а не влияние параметра)")
        lines.append("")
        lines.extend(_table(
            [
                [
                    str(item.get("rank", "-")),
                    str(item.get("model")),
                    str(item.get("feature")),
                    f"{item.get('best_value')} → {_fmt(item.get('best_mean'))}",
                    f"{item.get('worst_value')} → {_fmt(item.get('worst_mean'))}",
                    _fmt(item.get("spread")),
                ]
                for item in varied
            ],
            ["место", "модель", "настройка", "лучшее", "худшее", "разброс"],
        ))
    elif not payload.get("settings_recorded"):
        # Различие существенно: «настроек не записано» и «настройки не
        # менялись» — это разные диагнозы, и свести их к одному
        # значит скрыть, откуда взялся пробел в выводах.
        lines.append(
            "    нет: применённые настройки в манифестах этих батчей не "
            "записаны. Они фиксируются начиная со следующего прогона."
        )
    else:
        lines.append(
            "    нет: ни одна настройка не менялась внутри своей модели, "
            "сравнивать нечего. Оценить влияние можно, изменив конфигурацию."
        )
    lines.append("")

    not_evaluated = (payload.get("settings_influence") or {}).get(
        "not_evaluated"
    ) or []
    if not_evaluated:
        lines.append("  Не оценено:")
        for item in not_evaluated:
            lines.append(f"    · {item.get('feature')} — {item.get('note')}")
        lines.append("")

    constant = (payload.get("settings_influence") or {}).get("constant") or []
    if constant:
        lines.append(
            "  Настройки, не проверенные на влияние (были постоянны): "
            + ", ".join(sorted(str(item.get("feature")) for item in constant))
        )
        lines.append(
            "    Оценить их можно, изменив конфигурацию и повторив прогон: "
            "по постоянному признаку сравнивать нечего."
        )
        lines.append("")

    conditions = payload.get("condition_influence") or []
    if conditions:
        lines.append("  Связь качества с условиями обучения (ρ Спирмена):")
        lines.extend(_table(
            [
                [
                    str(item.get("feature")),
                    f"{item.get('spearman'):+.3f}" if item.get("spearman") is not None else "н/д",
                    str(item.get("n")),
                ]
                for item in conditions
            ],
            ["признак", "ρ", "наблюдений"],
        ))
        lines.append("")

    calibration = payload.get("calibration") or []
    if calibration:
        lines.append("  Калибровка: доля предсказанных положительных против фактической")
        lines.extend(_table(
            [
                [
                    str(item.get("model")),
                    f"{item.get('predicted_rate', 0) * 100:.1f} %",
                    f"{item.get('actual_rate', 0) * 100:.1f} %",
                    f"×{item.get('ratio')}" if item.get("ratio") else "н/д",
                ]
                for item in calibration
            ],
            ["модель", "предсказано", "фактически", "превышение"],
        ))
        lines.append(
            "    Превышение означает систематическое завышение: модель "
            "называет положительным слишком много полисов. Это ограничивает"
        )
        lines.append(
            "    точность положительного класса сильнее, чем показывает f1."
        )
        lines.append("")

    return lines


def _section_sweep(config: Config) -> list[str]:
    """Результаты перебора вариантов предобработки (3.b.i)."""
    payload = load_json(config.reports_dir / "preprocessing_sweep.json")
    if not payload:
        return []

    lines = ["-" * 78, "11. ПЕРЕБОР ВАРИАНТОВ ПРЕДОБРАБОТКИ", "-" * 78]
    lines.append("")
    lines.append(
        f"  Батч {payload.get('batch_idx')}, модель {payload.get('model')!r}, "
        f"метрика {payload.get('metric')!r}"
    )
    lines.append(
        "  Ограничение: выбор сделан на одной модели и одном батче. Дерево, лес"
    )
    lines.append(
        "  и нейросеть реагируют на масштаб и импутацию иначе, поэтому перенос"
    )
    lines.append("  выбора на них автоматически не выполняется.")
    lines.append("")

    usable = [item for item in payload.get("results", []) if item.get("status") == "ok"]
    if usable:
        lines.extend(_table(
            [
                [
                    str(item.get("name")),
                    _fmt(item.get("f1")),
                    _fmt(item.get("roc_auc")),
                    str(item.get("n_features")),
                    f"{item.get('seconds', 0):.2f}",
                ]
                for item in usable
            ],
            ["вариант", "f1", "roc_auc", "признаков", "секунд"],
        ))

    failed = [item for item in payload.get("results", []) if item.get("status") != "ok"]
    for item in failed:
        lines.append(f"  {item.get('name')}: не построился — {item.get('error')}")

    best = payload.get("best") or {}
    if best.get("status") == "none":
        lines.append("")
        lines.append(
            f"  Ни один вариант не построился, остаётся базовая конфигурация."
        )
    else:
        lines.append("")
        lines.append(
            f"  Победил {best.get('name')} "
            f"({payload.get('metric')} = {_fmt(best.get(payload.get('metric')))})"
        )
    lines.append("")
    return lines


def _section_performance(manifests: list[dict[str, Any]]) -> list[str]:
    """Производительность: время обучения и потребление памяти."""
    lines = ["-" * 78, "9. ПРОИЗВОДИТЕЛЬНОСТЬ", "-" * 78]
    if not manifests:
        lines.append("Нет данных о производительности.")
        lines.append("")
        return lines

    totals: dict[str, float] = {}
    for item in manifests:
        for stage, value in item.get("performance", {}).get("durations", {}).items():
            totals[stage] = totals.get(stage, 0.0) + float(value)

    rows = [[stage, f"{value:.3f}", format_duration(value)] for stage, value in totals.items()]
    rows.sort(key=lambda row: -float(row[1]))
    lines.extend(_table(rows, ["этап", "секунд (сумма)", "человекочитаемо"]))
    lines.append("")

    memories = [
        item.get("performance", {}).get("peak_rss_mb")
        for item in manifests
        if item.get("performance", {}).get("peak_rss_mb") is not None
    ]
    if memories:
        lines.append(
            f"  Пиковое потребление памяти процесса (RSS): {max(memories):.1f} МБ"
        )
    else:
        lines.append("  Пиковое потребление памяти (RSS): нет данных для этой платформы")
    lines.append("")

    training = manifests[-1].get("training", {})
    if training.get("durations"):
        lines.append("  Время обучения моделей в последнем батче (с):")
        for name, value in training["durations"].items():
            lines.append(f"    {name:<8} {value}")
        lines.append("")

    preprocessor = manifests[-1].get("preprocessor", {})
    if preprocessor.get("n_features"):
        lines.append(
            f"  Размерность признакового пространства: {preprocessor['n_features']} "
            f"(отпечаток {preprocessor.get('fingerprint_hash', 'н/д')})"
        )
    lines.append("")
    return lines
