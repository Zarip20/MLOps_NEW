"""Сборка слоя вывода: конкретные представления за портами (7.b.iv).

Этот модуль — «сшитый» адаптер: он переводит вызов порта в вызовы
существующих модулей отчёта, дашборда и публикации. Зависимости
развёрнуты в одну сторону: `views` знает про `report`, `dashboard`,
`publish` и `meta`, но не про конвейер.

Представлений четыре, а не три: анализ Meta Learning сам по себе
представлением не является — это вычисление. Но его результат должен
быть готов **до** сборки отчёта и дашборда, которые его читают, иначе
оба артефакта вышли бы без раздела об анализе и выглядели бы полными,
будучи неполными. Порядок регистрации это фиксирует.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.dashboard import build_dashboard
from src.meta import analyse as analyse_meta
from src.publish import build_site
from src.report import build_report
from src.utils import read_artifacts, save_json

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Всё, что нужно представлениям.

    Собирается в одном месте и передаётся дальше как есть: так
    представления не достают то, чего им не дали, и не знают, откуда
    у конвейера те или иные данные.
    """

    config: Any
    state: Any
    watch: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    def manifests(self) -> list[dict[str, Any]]:
        return read_artifacts(self.config.metadata_dir, "run_manifest")


@dataclass
class MetaAnalysisView:
    """Считает Meta Learning и кладёт результат в отчёт.

    Идёт первым: остальные представления читают `reports/meta.json`,
    и собранные до него они вышли бы без раздела об анализе.
    """

    name: str = "meta_learning"

    def render(self, context: Any) -> Path:
        payload = analyse_meta(
            context.manifests(),
            metric=context.config.meta_metric,
            min_runs=context.config.meta_min_runs,
        )
        path = save_json(context.config.reports_dir / "meta.json", payload)
        logger.info(
            "Meta Learning: батчей %d, прогонов %d, выводов %d",
            payload.get("n_batches", 0), payload.get("n_runs", 0),
            len(payload.get("findings") or []),
        )
        return path


@dataclass
class TextReportView:
    """Текстовый отчёт для чтения человеком и для сдачи."""

    name: str = "text_report"

    def render(self, context: Any) -> Path:
        return build_report(context.config, context.state)


@dataclass
class HtmlDashboardView:
    """Дашборд истории обучения."""

    name: str = "html_dashboard"

    def render(self, context: Any) -> Path:
        return build_dashboard(
            context.config.reports_dir, context.config.metadata_dir
        )


@dataclass
class SiteView:
    """Статический сайт для GitHub Pages.

    Дашборд не пересобирается: к этому моменту его уже построил
    `HtmlDashboardView`, а повторная сборка дала бы ту же страницу
    второе место в логе и лишние секунды работы.
    """

    name: str = "site"

    def render(self, context: Any) -> Path:
        result = build_site(context.config, refresh_dashboard=False)
        context.extras["site"] = result
        return result.root / "index.html"


def default_registry() -> Any:
    """Набор представлений по умолчанию, в правильном порядке."""
    from src.ports import ViewRegistry

    registry = ViewRegistry()
    registry.register(MetaAnalysisView())
    registry.register(TextReportView())
    registry.register(HtmlDashboardView())
    registry.register(SiteView())
    return registry
