"""Порты представлений: контракт между оркестратором и слоем вывода (7.b.iv).

Требование: «архитектурные паттерны (MVC / Ports & Adapters): тонкий
`run.py` → `src/pipeline.py` (контроллер), доменные модули (модель),
представления (отчёты, дашборд)». Разделение в проекте фактически есть,
но оно не выражено в типах: `pipeline.py` вызывал `build_report` и
`build_dashboard` напрямую, и добавить третий вид вывода означало бы
править оркестратор.

Здесь вводится порт — узкий контракт «что-то, что умеет построить
артефакт» — и реестр представлений. Оркестратор знает только порт и
работает с любым числом представлений, не зная их имён.

**Что это даёт и чего не даёт.** Даёт: новое представление добавляется
регистрацией одной строкой, а не правкой оркестратора; и — важнее —
позволяет подставить подставной объект в тестах и проверить порядок и
состав вывода, не строя настоящий дашборд. Не даёт: независимости от
конкретных модулей отчёта и дашборда внутри них самих. Для этого есть
`views.py`, где они и собраны.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class View(Protocol):
    """Порт представления: умеет построить артефакт и сказать, куда."""

    #: Короткое имя для логов и отчёта.
    name: str

    def render(self, context: Any) -> Path:
        """Построить артефакт.

        Args:
            context: всё, что нужно представлению. Конкретный тип
                оставлен намеренно неуказанным: у представлений разные
                потребности, и общего узкого контекста для них не
                существует. Порт проверяет наличие метода, а не форму
                аргументов.
        """
        ...


@dataclass
class RenderResult:
    """Что получилось при выводе одного представления."""

    view: str
    path: Path | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ViewRegistry:
    """Список представлений, которые строит оркестратор.

    Порядок регистрации значим: текстовый отчёт читает результаты
    остальных, поэтому он идёт не последним, а так, как зарегистрирован.
    """

    views: list[View] = field(default_factory=list)

    def register(self, view: View) -> View:
        """Добавить представление и вернуть его же — для цепочки вызовов."""
        if not getattr(view, "name", None):
            raise ValueError("представление должно иметь непустое имя")
        if not callable(getattr(view, "render", None)):
            raise TypeError(
                f"{type(view).__name__} не является представлением: "
                f"нет метода render"
            )
        self.views.append(view)
        return view

    def render_all(
        self, context: Any, only: Sequence[str] | None = None
    ) -> list[RenderResult]:
        """Построить представления.

        Args:
            context: всё, что нужно представлениям.
            only: имена тех, которые нужно построить. `None` — все.
                Список нужен для режима «только сайт»: переписывать
                текстовый отчёт при публикации незачем, и в логе он
                появлялся бы дважды за один запуск.

        Сбой одного не отменяет остальных: текстовый отчёт не должен
        пропадать из-за того, что дашборд не собрался. Результаты
        собираются и возвращаются, а ошибка пишется в лог и попадает в
        возвращаемый список — вызывающий код сам решит, что с ней делать.
        """
        wanted = set(only) if only is not None else None
        results: list[RenderResult] = []
        for view in self.views:
            name = getattr(view, "name", type(view).__name__)
            if wanted is not None and name not in wanted:
                continue
            try:
                results.append(RenderResult(view=name, path=view.render(context)))
            except Exception as error:  # noqa: BLE001
                logger.warning("Представление %s не построило вывод: %s", name, error)
                results.append(RenderResult(view=name, error=str(error)))
        return results

    def names(self) -> list[str]:
        return [getattr(view, "name", type(view).__name__) for view in self.views]
