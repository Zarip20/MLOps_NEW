"""Дашборд истории обучения (7.b.i, задание 2 — 2.b.iii).

Собственный генератор HTML на стандартной библиотеке, без Dash, Streamlit
и Plotly (решение Р-4). Причина не только в запрете сторонних MLOps-инструментов,
но и в практической: дашборд должен собираться в артефакт CI без
браузера и без сети.

Что показывается:

* сводка прогона: батчи, версии моделей, конфигурация, окружение;
* динамика качества данных (полнота, доля положительного класса);
* динамика метрик всех моделей с выделением продуктовой;
* дрейф: PSI и KS по батчам, статус, пороги;
* ассоциативные правила и найденные по ним нарушения;
* важнейшие признаки по моделям и согласованность между моделями;
* производительность: время по этапам и память.

Все данные берутся из JSON-артефактов, которые конвейер уже пишет, —
дашборд ничего не пересчитывает и не требует повторного обучения.
"""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

#: Палитра. Достаточный контраст на светлом фоне, различима при
#: печати и при grayscale-воспроизведении.
PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]
STATUS_COLORS = {"ok": "#059669", "warn": "#d97706", "alert": "#dc2626", "unknown": "#6b7280"}


def build_dashboard(
    reports_dir: str | Path,
    metadata_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Собрать HTML-дашборд по накопленным артефактам.

    Args:
        reports_dir: каталог с текстовыми и JSON-отчётами.
        metadata_dir: каталог с метаданными батчей.
        output_path: куда сохранить. По умолчанию `reports/dashboard.html`.

    Returns:
        Путь к сохранённому файлу.
    """
    reports_dir = Path(reports_dir)
    metadata_dir = Path(metadata_dir)

    quality = _read_all(metadata_dir, "quality")
    metrics = _read_all(metadata_dir, "model_metrics")
    manifests = _read_all(metadata_dir, "run_manifest")

    if not manifests:
        logger.warning("Нет манифестов — дашборд будет пустым")

    target = Path(output_path) if output_path else reports_dir / "dashboard.html"
    target.parent.mkdir(parents=True, exist_ok=True)

    document = _render(quality, metrics, manifests, reports_dir)
    target.write_text(document, encoding="utf-8")
    logger.info("Дашборд сохранён: %s (%d байт)", target.name, len(document))
    return target


def _read_all(directory: Path, prefix: str) -> list[dict[str, Any]]:
    """Прочитать все JSON-артефакты с заданным префиксом по индексу батча."""
    if not directory.is_dir():
        return []
    items: list[tuple[int, dict[str, Any]]] = []
    for path in directory.glob(f"{prefix}_*.json"):
        stem = path.stem.rsplit("_", 1)[-1]
        try:
            index = int(stem)
        except ValueError:
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict):
            items.append((index, payload))
    return [payload for _, payload in sorted(items, key=lambda pair: pair[0])]


# ---------------------------------------------------------------------------
# Рендеринг
# ---------------------------------------------------------------------------


def _render(
    quality: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    reports_dir: Path,
) -> str:
    summary = _collect_summary(quality, metrics, manifests)

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="ru"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>MLOps — дашборд обучения ({html.escape(summary['generated'])})</title>",
        "<style>" + _STYLES + "</style>",
        "</head><body>",
        "<header>",
        "<h1>MLOps-конвейер: история обучения</h1>",
        f"<p class='muted'>Сформирован {html.escape(summary['generated'])} · "
        f"конфигурация <code>{html.escape(summary['config_hash'])}</code> · "
        f"обработано батчей {summary['batches_processed']} из {summary['batches_total']}</p>",
        "</header>",
    ]

    parts.append(_section_cards(summary))
    parts.append(_section_quality(quality, summary))
    parts.append(_section_metrics(metrics, summary))
    parts.append(_section_drift(manifests))
    parts.append(_section_target(quality, summary))
    parts.append(_section_rules(summary))
    parts.append(_section_features(summary))
    parts.append(_section_registry(summary))
    parts.append(_section_performance(manifests))
    parts.append(_section_reports(reports_dir))

    parts.append(
        "<footer><p class='muted'>Дашборд собран из артефактов "
        "<code>data/metadata/*.json</code>. Генерируется средствами "
        "стандартной библиотеки Python.</p></footer>"
    )
    parts.append("</body></html>")
    return "\n".join(part for part in parts if part)


def _collect_summary(
    quality: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    """Свести данные из артефактов в единую структуру."""
    summary: dict[str, Any] = {
        "generated": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "batches_processed": len(manifests),
        "batches_total": len(manifests),
        "config_hash": "",
        "models": [],
        "series": {},
    }

    if manifests:
        config = manifests[-1].get("config", {})
        summary["config_hash"] = config.get("config_hash", "")
        summary["validation"] = config.get("validation", {})
        summary["environment"] = manifests[-1].get("environment", {})
        summary["feature_count"] = manifests[-1].get("preprocessor", {}).get("n_features")
        summary["stored_rows"] = manifests[-1].get("training_store", {}).get("rows")

    # Имена моделей: берём из метрик последнего батча
    for item in metrics:
        for key, value in item.items():
            if key in ("batch_idx", "batch") or not isinstance(value, dict):
                continue
            if key not in summary["models"]:
                summary["models"].append(key)
    summary["models"].sort()

    # Ряды для графиков
    indices = [item.get("batch_idx", index) for index, item in enumerate(metrics)]
    summary["indices"] = indices
    for name in summary["models"]:
        summary["series"][name] = [
            item.get(name, {}).get("f1") for item in metrics
        ]
        summary["series"][f"{name}_roc"] = [
            item.get(name, {}).get("roc_auc") for item in metrics
        ]

    summary["completeness"] = [
        item.get("completeness", {}).get("total_missing_ratio") for item in quality
    ]
    summary["positive_rate"] = [
        (item.get("target") or {}).get("positive_rate") for item in quality
    ]
    summary["rows_removed"] = [
        (item.get("cleaning") or {}).get("rows_removed") for item in quality
    ]
    summary["indices_quality"] = [item.get("batch_idx", i) for i, item in enumerate(quality)]

    summary["rules"] = manifests[-1].get("association", {}) if manifests else {}

    # Реестр моделей: берём из последнего манифеста, если он его содержит
    summary["registry"] = manifests[-1].get("registry", {}) if manifests else {}

    # Важнейшие признаки из последнего батча
    summary["explanation"] = manifests[-1].get("explanation", {}) if manifests else {}

    return summary


def _section_cards(summary: dict[str, Any]) -> str:
    production = (summary.get("registry") or {}).get("production") or {}
    environment = summary.get("environment", {})
    validation = summary.get("validation", {})

    cards = [
        ("Обработано батчей", f"{summary['batches_processed']}"),
        ("Размерность признаков", summary.get("feature_count") or "—"),
        ("Накоплено строк", summary.get("stored_rows") or "—"),
        (
            "Продуктовая модель",
            f"{production.get('model', '—')} {production.get('version', '')}".strip() or "—",
        ),
        (
            "Продуктовая f1",
            _fmt(production.get("f1")),
        ),
        (
            "Версий в реестре",
            (summary.get("registry") or {}).get("n_versions", "—"),
        ),
        ("Проверка качества", validation.get("test_size", "—")),
        ("Python / sklearn", f"{environment.get('python', '—')} / {environment.get('scikit_learn', '—')}"),
    ]

    cells = "".join(
        f"<div class='card'><div class='card-label'>{html.escape(str(label))}</div>"
        f"<div class='card-value'>{html.escape(str(value))}</div></div>"
        for label, value in cards
    )
    return f"<section><h2>Сводка</h2><div class='cards'>{cells}</div></section>"


def _section_quality(quality: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if not quality:
        return ""
    return (
        "<section><h2>Качество данных</h2>"
        + _chart(
            summary["indices_quality"],
            [
                ("Полнота признаков", summary["completeness"], "#2563eb"),
                ("Доля положительного класса", summary["positive_rate"], "#dc2626"),
                ("Удалено строк очисткой", summary["rows_removed"], "#d97706"),
            ],
            note="Падение доли положительного класса с 11.0 % до 2.4 % — дрейф "
                 "целевой метки, а не деградация модели.",
        )
        + "</section>"
    )


def _section_metrics(metrics: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if not metrics:
        return ""
    production = (summary.get("registry") or {}).get("production") or {}
    production_model = production.get("model")

    series = []
    for index, name in enumerate(summary["models"]):
        colour = PALETTE[index % len(PALETTE)]
        label = f"{name} (f1)"
        if name == production_model:
            label += " — продуктовая"
        series.append((label, summary["series"][name], colour))

    return (
        "<section><h2>Качество моделей (f1 на отложенной валидации)</h2>"
        + _chart(summary["indices"], series, note="Отбор лучшей модели идёт по f1.")
        + "</section>"
    )


def _section_drift(manifests: list[dict[str, Any]]) -> str:
    if not manifests:
        return ""

    indices = [item.get("batch_idx", index) for index, item in enumerate(manifests)]
    psi_values = [
        (item.get("drift") or {}).get("max_psi") for item in manifests
    ]
    ks_values = [
        (item.get("drift") or {}).get("max_ks") for item in manifests
    ]

    thresholds = {}
    for item in manifests:
        drift = item.get("drift") or {}
        if drift.get("thresholds"):
            thresholds = drift["thresholds"]
            break

    rows: list[str] = []
    for item in manifests:
        drift = item.get("drift") or {}
        status = drift.get("status", "unknown")
        target = drift.get("target") or {}
        rows.append(
            "<tr>"
            f"<td>{item.get('batch_idx', '')}</td>"
            f"<td><span class='badge' style='background:{STATUS_COLORS.get(status, '#6b7280')}'>"
            f"{html.escape(status)}</span></td>"
            f"<td>{_fmt(drift.get('max_psi'))}</td>"
            f"<td>{_fmt(drift.get('max_ks'))}</td>"
            f"<td>{_fmt(target.get('psi'))}</td>"
            f"<td>{_fmt(target.get('actual_positive_rate'))}</td>"
            "</tr>"
        )

    limit = (thresholds.get("psi_alert") or 0.25)
    series = [
        ("Максимальный PSI по признакам", psi_values, "#d97706"),
        ("Максимальный KS", ks_values, "#7c3aed"),
    ]

    return (
        "<section><h2>Дрейф данных</h2>"
        + _chart(indices, series, thresholds=[limit], threshold_label="порог alert")
        + "<table><thead><tr><th>батч</th><th>статус</th><th>max PSI</th>"
          "<th>max KS</th><th>PSI метки</th><th>доля POS</th></tr></thead>"
        + "<tbody>" + "".join(rows) + "</tbody></table>"
        + "<p class='muted'>Дрейф не прерывает конвейер: он фиксируется в "
          "метаданных и учитывается гейтом качества при выборе продукта.</p>"
        "</section>"
    )


def _section_target(quality: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Отдельный акцент на дрейфе целевой метки.

    Это главная находка прогонов: без неё падение f1 с 0.475 до 0.221
    выглядит как «модель испортилась» и лечится переобучением, которое
    не помогает, потому что изменились данные, а не модель.
    """
    rates = [rate for rate in summary["positive_rate"] if rate is not None]
    if len(rates) < 2:
        return ""
    change = (rates[-1] / rates[0] - 1) * 100 if rates[0] else 0.0
    return (
        "<section><h2>Целевая метка: дрейф по времени</h2>"
        f"<div class='cards'>"
        f"<div class='card'><div class='card-label'>Доля POS, первый батч</div>"
        f"<div class='card-value'>{rates[0]:.4f}</div></div>"
        f"<div class='card'><div class='card-label'>Доля POS, последний батч</div>"
        f"<div class='card-value'>{rates[-1]:.4f}</div></div>"
        f"<div class='card'><div class='card-label'>Изменение</div>"
        f"<div class='card-value'>{change:+.1f} %</div></div>"
        "</div>"
        "<p class='muted'>Снижение доли страховых случаев почти в пять раз — "
        "объяснение падения f1. Модель обучена на более «рискованном» "
        "периоде и на последних батчах систематически завышает "
        "вероятность.</p></section>"
    )


def _section_rules(summary: dict[str, Any]) -> str:
    rules = summary.get("rules") or {}
    if not rules:
        return ""
    return (
        "<section><h2>Ассоциативные правила</h2>"
        f"<p class='muted'>Найдено правил: {rules.get('n_rules', 0)} · "
        f"средний lift {rules.get('mean_lift', '—')} · "
        f"средняя уверенность {rules.get('mean_confidence', '—')}</p>"
        "<p class='muted'>Отбор идёт по lift, а не по уверенности: при отборе "
        "по уверенности в выдачу попадают только правила в сторону "
        "мажоритного класса.</p></section>"
    )


def _section_features(summary: dict[str, Any]) -> str:
    explanation = summary.get("explanation") or {}
    by_model = explanation.get("by_model") or {}
    if not by_model:
        return ""

    blocks: list[str] = []
    for name, payload in by_model.items():
        features = payload.get("features") or []
        if not features:
            continue
        key = "coefficient" if "coefficient" in features[0] else "importance"
        rows = "".join(
            f"<tr><td>{html.escape(str(item['feature']))}</td>"
            f"<td class='num'>{_fmt(item.get(key))}</td>"
            f"<td>{html.escape(str(item.get('direction', item.get('share', ''))))}</td></tr>"
            for item in features[:10]
        )
        blocks.append(
            f"<h3>{html.escape(name)} — {html.escape(payload.get('method', ''))}</h3>"
            f"<p class='muted'>{html.escape(payload.get('interpretation', ''))}</p>"
            f"<table><thead><tr><th>признак</th><th>величина</th><th>интерпретация</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
        )

    consensus = explanation.get("consensus") or []
    consensus_rows = "".join(
        f"<li><code>{html.escape(str(item['feature']))}</code> — "
        f"{item['models_agreeing']} мод.</li>"
        for item in consensus[:8]
    )
    if consensus_rows:
        blocks.append(
            "<h3>Признаки, выделенные несколькими моделями</h3>"
            f"<ul class='consensus'>{consensus_rows}</ul>"
        )

    return "<section><h2>Интерпретация моделей</h2>" + "".join(blocks) + "</section>"


def _section_registry(summary: dict[str, Any]) -> str:
    registry = summary.get("registry") or {}
    versions = registry.get("versions") or []
    if not versions:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(str(item.get('version')))}</td>"
        f"<td>{html.escape(str(item.get('model')))}</td>"
        f"<td>{item.get('batch_idx', '')}</td>"
        f"<td class='num'>{_fmt(item.get('f1'))}</td>"
        f"<td>{html.escape(str(item.get('status', '')))}</td>"
        f"<td class='muted'>{html.escape('; '.join(item.get('reasons', []) or []))}</td></tr>"
        for item in versions[-15:]
    )
    return (
        "<section><h2>Реестр версий моделей</h2>"
        "<table><thead><tr><th>версия</th><th>модель</th><th>батч</th><th>f1</th>"
        "<th>статус</th><th>причины отказа</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></section>"
    )


def _section_performance(manifests: list[dict[str, Any]]) -> str:
    if not manifests:
        return ""
    totals: dict[str, float] = {}
    memory: list[float] = []
    for item in manifests:
        performance = item.get("performance") or {}
        for stage, value in (performance.get("durations") or {}).items():
            totals[stage] = totals.get(stage, 0.0) + float(value)
        rss = performance.get("peak_rss_mb")
        if rss:
            memory.append(float(rss))

    rows = "".join(
        f"<tr><td>{html.escape(stage)}</td><td class='num'>{value:.2f}</td></tr>"
        for stage, value in sorted(totals.items(), key=lambda pair: -pair[1])
    )
    summary_memory = f"<p class='muted'>Пик памяти процесса: {max(memory):.1f} МБ</p>" if memory else ""
    return (
        "<section><h2>Производительность</h2>"
        "<table><thead><tr><th>этап</th><th>секунд суммарно</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>{summary_memory}</section>"
    )


def _section_reports(reports_dir: Path) -> str:
    files = sorted(reports_dir.glob("summary_*.txt"))[-5:] if reports_dir.is_dir() else []
    if not files:
        return ""
    items = "".join(
        f"<li><code>{html.escape(item.name)}</code></li>" for item in reversed(files)
    )
    return (
        "<section><h2>Текстовые отчёты</h2>"
        f"<ul class='files'>{items}</ul>"
        "<p class='muted'>Полные отчёты выгружаются из CI отдельным "
        "артефактом.</p></section>"
    )


# ---------------------------------------------------------------------------
# Графики без внешних библиотек
# ---------------------------------------------------------------------------


def _chart(
    indices: Sequence[Any],
    series: Sequence[tuple[str, Sequence[Any], str]],
    thresholds: Sequence[float] | None = None,
    threshold_label: str = "",
    width: int = 1000,
    height: int = 260,
    note: str = "",
) -> str:
    """Нарисовать линейный график как SVG средствами стандартной библиотеки."""
    thresholds = list(thresholds or [])
    usable = [
        (label, [None if value is None else float(value) for value in values], colour)
        for label, values, colour in series
    ]
    finite = [value for _, values, _ in usable for value in values if value is not None]
    finite += list(thresholds)

    if not finite or not indices:
        return "<p class='muted'>Недостаточно данных для графика.</p>"

    padding_left, padding_right, padding_top, padding_bottom = 62, 16, 14, 34
    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom

    minimum, maximum = min(finite), max(finite)
    if maximum - minimum < 1e-9:
        maximum = minimum + 1.0
    span = maximum - minimum
    minimum -= span * 0.05
    maximum += span * 0.05
    span = maximum - minimum

    count = max(len(indices), 1)

    def x_position(index: int) -> float:
        return padding_left + (plot_width * index / max(count - 1, 1))

    def y_position(value: float) -> float:
        return padding_top + plot_height - (plot_height * (value - minimum) / span)

    parts: list[str] = [
        f"<svg viewBox='0 0 {width} {height}' class='chart' role='img' "
        f"preserveAspectRatio='xMidYMid meet'>"
    ]

    # Сетка и подписи оси Y
    for step in range(5):
        value = minimum + span * step / 4
        y = y_position(value)
        parts.append(
            f"<line x1='{padding_left}' y1='{y:.1f}' x2='{width - padding_right}' "
            f"y2='{y:.1f}' class='grid'/>"
            f"<text x='{padding_left - 8}' y='{y + 4:.1f}' class='axis' "
            f"text-anchor='end'>{_short(value)}</text>"
        )

    # Пороговые линии
    for threshold in thresholds:
        if not (minimum <= threshold <= maximum):
            continue
        y = y_position(threshold)
        parts.append(
            f"<line x1='{padding_left}' y1='{y:.1f}' x2='{width - padding_right}' "
            f"y2='{y:.1f}' class='threshold'/>"
            f"<text x='{width - padding_right - 4}' y='{y - 4:.1f}' class='axis' "
            f"text-anchor='end'>{html.escape(threshold_label)} = {threshold}</text>"
        )

    # Подписи оси X: не чаще, чем раз в 6 батчей
    for index, label in enumerate(indices):
        if index % 6 == 0 or index == count - 1:
            parts.append(
                f"<text x='{x_position(index):.1f}' y='{height - 12}' class='axis' "
                f"text-anchor='middle'>{html.escape(str(label))}</text>"
            )

    # Серии
    for label, values, colour in usable:
        points: list[str] = []
        previous: tuple[float, float] | None = None
        for index, value in enumerate(values):
            if value is None:
                # Разрыв в данных: линия не проводится через пропуск
                previous = None
                continue
            x, y = x_position(index), y_position(value)
            if previous is not None:
                points.append(f"{previous[0]:.1f},{previous[1]:.1f}")
            points.append(f"{x:.1f},{y:.1f}")
            previous = (x, y)
        if len(points) >= 2:
            parts.append(
                f"<polyline points='{' '.join(points)}' fill='none' "
                f"stroke='{colour}' stroke-width='2'/>"
            )
        for index, value in enumerate(values):
            if value is not None:
                x, y = x_position(index), y_position(value)
                parts.append(
                    f"<circle cx='{x:.1f}' cy='{y:.1f}' r='2.5' fill='{colour}'>"
                    f"<title>{html.escape(label)}: {_fmt(value)}</title></circle>"
                )
    parts.append("</svg>")

    legend = "".join(
        f"<span class='legend-item'><i style='background:{colour}'></i>"
        f"{html.escape(label)}</span>"
        for label, _, colour in usable
    )
    note_html = f"<p class='muted'>{html.escape(note)}</p>" if note else ""
    return f"<div class='chart-wrap'>{''.join(parts)}<div class='legend'>{legend}</div>{note_html}</div>"


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return html.escape(str(value))


def _short(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1000:
        return f"{value:.0f}"
    if absolute >= 10:
        return f"{value:.0f}"
    if absolute >= 1:
        return f"{value:.1f}"
    if absolute >= 0.01:
        return f"{value:.2f}"
    return f"{value:.3f}"


_STYLES = """
:root{--ink:#111827;--muted:#6b7280;--line:#e5e7eb;--bg:#ffffff;--card:#f9fafb}
*{box-sizing:border-box}
body{margin:0;padding:0 24px 48px;font:14px/1.55 -apple-system,BlinkMacSystemFont,
 'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;color:var(--ink);background:var(--bg)}
header{padding:28px 0 12px;border-bottom:1px solid var(--line);margin-bottom:24px}
h1{font-size:22px;margin:0 0 6px}
h2{font-size:17px;margin:32px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:14px;margin:20px 0 8px}
p{margin:8px 0}
.muted{color:var(--muted);font-size:13px}
code{background:var(--card);padding:1px 5px;border-radius:3px;font-size:12px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;
 padding:12px 16px;min-width:150px;flex:1 1 150px}
.card-label{font-size:11px;text-transform:uppercase;letter-spacing:.04em;
 color:var(--muted);margin-bottom:4px}
.card-value{font-size:19px;font-weight:600;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;margin:10px 0;font-size:13px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);
 vertical-align:top}
th{font-weight:600;font-size:12px;color:var(--muted);text-transform:uppercase;
 letter-spacing:.03em}
td.num{text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:var(--card)}
.badge{display:inline-block;color:#fff;padding:1px 8px;border-radius:9px;
 font-size:11px;font-weight:600}
.chart-wrap{margin:10px 0 16px}
.chart{width:100%;height:auto;display:block}
.grid{stroke:var(--line);stroke-width:1}
.threshold{stroke:#dc2626;stroke-width:1;stroke-dasharray:5 4;opacity:.65}
.axis{font-size:10px;fill:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:6px;font-size:12px;
 color:var(--muted)}
.legend-item{display:inline-flex;align-items:center;gap:6px}
.legend-item i{width:12px;height:3px;border-radius:2px;display:inline-block}
ul.consensus,ul.files{margin:6px 0;padding-left:20px;font-size:13px}
footer{margin-top:40px;padding-top:14px;border-top:1px solid var(--line)}
"""
