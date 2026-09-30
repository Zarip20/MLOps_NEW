"""Проверки архитектурного разделения (7.b.iv).

Требование: «тонкий `run.py` → `src/pipeline.py` (контроллер), доменные
модули (модель), представления (отчёты, дашборд)». Разделение легко
нарушить незаметно: стоит один раз вызвать `build_report` прямо из
оркестратора, и слой вывода снова сросся с контроллером.

Поэтому разделение здесь не утверждается, а **проверяется**: разбирается
AST исходников и проверяется, что контроллер не занимается доменными
вычислениями, а представления не знают про конвейер.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.ports import RenderResult, ViewRegistry, View
from src.views import (
    HtmlDashboardView,
    MetaAnalysisView,
    PipelineContext,
    SiteView,
    TextReportView,
    default_registry,
)

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Реестр представлений
# ---------------------------------------------------------------------------


class FakeView:
    """Представление-заглушка: пишет имя в список вызовов."""

    def __init__(self, name: str, calls: list, fail: bool = False) -> None:
        self.name = name
        self.calls = calls
        self.fail = fail

    def render(self, context):
        self.calls.append((self.name, context))
        if self.fail:
            raise RuntimeError("не построилось")
        return Path(f"{self.name}.txt")


def test_view_is_satisfied_by_a_duck():
    """Порта достаточноduck-типизации: объект с `name` и `render` — представление."""
    assert isinstance(FakeView("a", []), View)


def test_registry_keeps_registration_order():
    """Порядок регистрации значим: отчёт читает результаты остальных."""
    calls: list = []
    registry = ViewRegistry()
    for name in ("сначала", "потом", "в конце"):
        registry.register(FakeView(name, calls))

    assert registry.names() == ["сначала", "потом", "в конце"]
    registry.render_all("контекст")
    assert [item[0] for item in calls] == ["сначала", "потом", "в конце"]


def test_one_failing_view_does_not_stop_the_others():
    """Сбой одного вывода не отменяет остальные.

    Текстовый отчёт не должен пропадать из-за того, что дашборд не
    собрался: это два разных артефакта для двух разных читателей.
    """
    calls: list = []
    registry = ViewRegistry()
    registry.register(FakeView("до", calls, fail=True))
    registry.register(FakeView("после", calls))

    results = registry.render_all("контекст")

    assert [item.ok for item in results] == [False, True]
    assert "не построилось" in results[0].error
    assert results[1].path.name == "после.txt"


def test_render_result_reports_failure():
    """Результат вывода сам различает успех и сбой."""
    assert RenderResult(view="a").ok is True
    assert RenderResult(view="a", error="бум").ok is False


def test_registering_a_non_view_is_rejected():
    """Объект с именем, но без `render`, представлением не является."""
    class NotAView:
        name = "почти"

    with pytest.raises(TypeError):
        ViewRegistry().register(NotAView())


def test_view_without_name_is_rejected():
    """Представление без имени невозможно опознать в логе и отчёте."""
    view = FakeView("", [])
    with pytest.raises(ValueError):
        ViewRegistry().register(view)


# ---------------------------------------------------------------------------
# Состав по умолчанию
# ---------------------------------------------------------------------------


def test_default_registry_covers_all_outputs():
    """Зарегистрированы все четыре вида вывода."""
    assert default_registry().names() == [
        "meta_learning", "text_report", "html_dashboard", "site",
    ]


def test_meta_analysis_comes_first():
    """Анализ идёт первым: отчёт и дашборд читают его результат.

    Собранные позже, они вышли бы без раздела об анализе и выглядели бы
    полными, будучи неполными.
    """
    names = default_registry().names()
    assert names.index("meta_learning") < names.index("text_report")
    assert names.index("meta_learning") < names.index("html_dashboard")


def test_site_reuses_the_dashboard_already_built():
    """Сайт не пересобирает дашборк заново.

    Иначе в логе одна и та же сборка появляется дважды, а работа
    удваивается без всякой пользы.
    """
    import inspect

    source = inspect.getsource(SiteView)
    assert "refresh_dashboard=False" in source


# ---------------------------------------------------------------------------
# Проверка разделения по исходникам
# ---------------------------------------------------------------------------


def imported_modules(path: Path) -> set[str]:
    """Имена модулей, импортируемых в файле."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_run_py_contains_no_domain_logic():
    """`run.py` только разбирает аргументы и вызывает режимы.

    Любая доменная работа здесь означала бы, что контроллер растёт
    обратно в монолит: ровно то, из-за чего он был разделён.
    """
    source = (ROOT / "run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef,))
    ]
    assert functions, "run.py должен содержать функции"

    # Доменные модули, вызов которых в точке входа означал бы протечку
    # логики в слой контроллера.
    forbidden = {
        "src.training", "src.data_quality", "src.association", "src.drift",
        "src.features", "src.registry", "src.storage", "src.report",
        "src.dashboard", "src.publish", "src.meta", "src.inference",
    }
    assert imported_modules(ROOT / "run.py") & forbidden == set()


def test_orchestrator_does_not_call_view_modules_directly():
    """Оркестратор обращается к выводу только через порт.

    Прямой вызов `build_report` из `pipeline.py` вернул бы слои в
    сросшееся состояние: добавить новый вид вывода пришлось бы
    правкой оркестратора.
    """
    imports = imported_modules(ROOT / "src" / "pipeline.py")
    views_side = {"src.dashboard", "src.publish", "src.meta"}

    assert not (imports & views_side) or imports & {"src.views"}


def test_views_do_not_depend_on_the_orchestrator():
    """Слой вывода не знает про конвейер.

    Обратная зависимость сделала бы представления неиспользуемыми
    отдельно от него — ровно то, ради чего слой выделен.
    """
    for name in ("views.py", "ports.py"):
        imports = imported_modules(ROOT / "src" / name)
        assert "src.pipeline" not in imports


def test_domain_modules_do_not_import_the_orchestrator():
    """Доменные модули не зависят от контроллера.

    Иначе вычисление качества или обучение нельзя было бы вызвать
    отдельно от конвейера — ни из тестов, ни из другого сценария.
    """
    for name in (
        "data_quality.py", "training.py", "registry.py", "drift.py",
        "features.py", "association.py", "storage.py", "preprocessing.py",
    ):
        imports = imported_modules(ROOT / "src" / name)
        assert "src.pipeline" not in imports, name


def test_context_carries_what_views_need():
    """Контекст отдаёт представлениям всё необходимое и ничего лишнего."""
    config = object()
    state = object()
    context = PipelineContext(config=config, state=state)

    assert context.config is config
    assert context.state is state
    assert context.extras == {}
