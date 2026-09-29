"""Проверки валидатора workflow-файлов.

Валидатор существует потому, что ошибка в YAML останавливает запуск
раньше, чем выполнится хоть одна строка кода, и сообщение об ошибке
указывает на строку файла, а не на смысл. Здесь проверяется, что он
замечает именно те расхождения, ради которых написан, и не ругается
на корректные файлы.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.check_workflow import check, load, main, triggers

VALID = """
name: Тестовый конвейер
on:
  push:
    branches: [ "main" ]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
      - run: python -m pytest
  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - run: python run.py -mode update
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "workflow.yml"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Корректные файлы
# ---------------------------------------------------------------------------


def test_valid_workflow_passes(tmp_path):
    """Корректный файл не вызывает замечаний."""
    assert check(write(tmp_path, VALID)) == []


def test_on_key_is_read_despite_yaml_boolean(tmp_path):
    """Ключ `on` YAML разбирает как True — триггеры всё равно видны.

    Без этой поправки валидатор объявил бы рабочий файл пустым и
    проверку пришлось бы отключить, потеряв её пользу.
    """
    data = load(write(tmp_path, VALID))
    assert triggers(data) == ["push"]


def test_needs_may_be_a_list(tmp_path):
    """Список зависимостей разбирается так же, как одиночная строка."""
    text = VALID.replace("    needs: test", "    needs: [test]")
    assert check(write(tmp_path, text)) == []


# ---------------------------------------------------------------------------
# Ошибки, которые должен замечать валидатор
# ---------------------------------------------------------------------------


def test_missing_needs_job_is_reported(tmp_path):
    """Зависимость от несуществующего job — ошибка, а не пустой прогон."""
    text = VALID.replace("needs: test", "needs: missing")
    problems = check(write(tmp_path, text))
    assert any("missing" in problem for problem in problems)


def test_job_without_steps_is_reported(tmp_path):
    """Job без шагов ничего не делает и выглядит как успешный."""
    text = """
on: [push]
jobs:
  train:
    runs-on: ubuntu-latest
    steps: []
"""
    problems = check(write(tmp_path, text))
    assert any("шагов" in problem for problem in problems)


def test_code_job_without_checkout_is_reported(tmp_path):
    """Job с `run`, но без checkout, работает в пустом каталоге.

    Самая неприятная поломка workflow: он зелёный, потому что скрипт
    не находит файлов, а падать там нечему.
    """
    text = """
on: [push]
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-python@v7
      - run: python run.py -mode update
"""
    problems = check(write(tmp_path, text))
    assert any("checkout" in problem for problem in problems)


def test_steps_without_actions_are_reported(tmp_path):
    """Шаг без `uses` и `run` — опечатка, которая тихо ничего не делает."""
    text = """
on: [push]
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - name: Заглушка
"""
    problems = check(write(tmp_path, text))
    assert any("без действий" in problem for problem in problems)


def test_missing_triggers_are_reported(tmp_path):
    """Workflow без триггеров не запустится никогда."""
    text = """
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
"""
    problems = check(write(tmp_path, text))
    assert any("триггер" in problem for problem in problems)


# ---------------------------------------------------------------------------
# Боевой файл проекта
# ---------------------------------------------------------------------------


def test_project_workflow_is_valid():
    """Боевой workflow проходит проверку.

    Именно эта проверка запускается в CI: workflow должен быть в
    порядке к моменту, когда его начнут выполнять.
    """
    path = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "main.yml"
    assert path.is_file(), f"нет файла {path}"
    assert check(path) == []


def test_project_workflow_uses_current_actions():
    """Версии action'ов зафиксированы на существующих major.

    Ссылка на несуществующий тег не ломает разбор файла — запуск
    падает уже в самом начале, до какой-либо полезной работы.
    """
    path = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "main.yml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    used: set[str] = set()
    for job in data["jobs"].values():
        for step in job.get("steps", []):
            if "uses" in step:
                used.add(str(step["uses"]))

    assert used, "в workflow не используются готовые action'ы"
    for action in used:
        assert "@v" in action, f"версия не зафиксирована: {action}"


def test_project_workflow_publishes_site():
    """Публикация дашборда подключена: сайт собирается и разворачивается.

    Требование задания 2 (2.b.iii) проверяется здесь же: без шага
    передачи сайта на развёртывание балл за онлайн-сервис не засчитывается.
    """
    path = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "main.yml"
    text = path.read_text(encoding="utf-8")
    assert "upload-pages-artifact" in text
    assert "deploy-pages" in text
    assert "pages: write" in text
    assert "id-token: write" in text


# ---------------------------------------------------------------------------
# Командный интерфейс
# ---------------------------------------------------------------------------


def test_cli_reports_success(tmp_path, capsys):
    """Успешная проверка печатает состав файла и возвращает 0."""
    assert main([str(write(tmp_path, VALID))]) == 0
    assert "в порядке" in capsys.readouterr().out


def test_cli_reports_failure(tmp_path, capsys):
    """Проблемный файл даёт ненулевой код возврата."""
    text = VALID.replace("needs: test", "needs: missing")
    assert main([str(write(tmp_path, text))]) == 1
    assert "ОШИБКА" in capsys.readouterr().out


def test_cli_reports_absent_file(tmp_path, capsys):
    """Отсутствующий файл — ошибка проверки, а не исключение."""
    assert main([str(tmp_path / "нет.yml")]) == 1
    assert "не найден" in capsys.readouterr().out


def test_cli_reports_unparsable_yaml(tmp_path, capsys):
    """Сломанный YAML даёт понятное сообщение, а не traceback."""
    path = write(tmp_path, "jobs: [не закрыто\n")
    assert main([str(path)]) == 1
    assert "не разобран" in capsys.readouterr().out
