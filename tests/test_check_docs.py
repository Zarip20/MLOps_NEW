"""Проверки самой проверки документации.

Смысл модуля — ловить расхождения, поэтому он не должен ни пропускать
настоящие, ни ругаться на отсутствие того, чего ещё нет. Обе крайности
проверяются здесь явно: первая обесценивает проверку, вторая приучает
её игнорировать.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests import check_docs


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Минимальный проект в отдельном каталоге."""
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "doc").mkdir()
    (root / "run.py").write_text("", encoding="utf-8")
    (root / "config.yaml").write_text("run: {}\n", encoding="utf-8")
    (root / "README.md").write_text("", encoding="utf-8")
    (root / "AGENTS.md").write_text("", encoding="utf-8")
    (root / "doc" / "task.md").write_text("", encoding="utf-8")
    (root / "doc" / "grade.md").write_text("", encoding="utf-8")
    (root / "doc" / "presentation.md").write_text("", encoding="utf-8")
    (root / "doc" / "github_actions.md").write_text("", encoding="utf-8")

    test_file = root / "tests" / "test_sample.py"
    test_file.write_text(
        "def test_one():\n    pass\n\n\ndef test_two():\n    pass\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_docs, "ROOT", root)
    # Число подставляется настоящее по величине: проверка ищет числа
    # из двух и более цифр, и «2» в документации просто невозможно.
    monkeypatch.setattr(check_docs, "count_tests", lambda: 306)
    return root


def write(root: Path, name: str, text: str) -> None:
    (root / name).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Итоговое число тестов
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (1, "тест"), (21, "тест"), (321, "тест"), (1021, "тест"),
        (2, "теста"), (22, "теста"), (222, "теста"),
        (5, "тестов"), (11, "тестов"), (15, "тестов"),
        (112, "тестов"), (111, "тестов"), (114, "тестов"),
    ],
)
def test_correct_form_follows_russian_rules(number, expected):
    """Форма вычисляется, а не набирается вручную.

    Ошибка «321 тестов» повторяется при каждом обновлении счётчика,
    и замечание проверки должно её исправлять само, а не требовать
    вспомнить правило.
    """
    assert check_docs.correct_form(number) == expected


def test_problem_message_suggests_the_right_form(project, monkeypatch):
    """Замечание содержит готовую форму для замены.

    Без неё пришлось бы вспоминать, что после 321 идёт «тест», а не
    «тестов», — а это как раз та ошибка, которая повторяется при
    каждом обновлении счётчика.
    """
    monkeypatch.setattr(check_docs, "count_tests", lambda: 321)
    write(project, "README.md", "| `pytest tests -q` | 302 тестов |\n")

    problems = check_docs.check_test_count()

    assert "321 тест)" in problems[0]
    assert "302 тестов" in problems[0]


# ---------------------------------------------------------------------------
# Исправление
# ---------------------------------------------------------------------------


def test_fix_rewrites_stale_totals(project, monkeypatch):
    """Устаревшее число заменяется на фактическое вместе со склонением."""
    monkeypatch.setattr(check_docs, "count_tests", lambda: 321)
    write(
        project, "README.md",
        "| `pytest tests -q` | 306 тест |\n"
        "├── tests/            # 306 тест\n",
    )

    fixed = check_docs.fix_totals()

    text = (project / "README.md").read_text(encoding="utf-8")
    assert "321 тест" in text
    assert "306" not in text
    assert fixed == ["README.md:1", "README.md:2"]
    assert check_docs.check_test_count() == []


def test_fix_chooses_the_right_grammatical_form(project, monkeypatch):
    """После 321 идёт «тест», после 322 — «теста», после 325 — «тестов»."""
    for total, expected in ((321, "321 тест"), (322, "322 теста"), (325, "325 тестов")):
        monkeypatch.setattr(check_docs, "count_tests", lambda value=total: value)
        write(project, "README.md", "| `pytest tests -q` | 1 тест |\n")
        check_docs.fix_totals()
        assert expected in (project / "README.md").read_text(encoding="utf-8")


def test_fix_leaves_per_file_counts_alone(project, monkeypatch):
    """Число тестов одного файла не переписывается.

    «18 тестов» про `test_collector.py` — верное утверждение, и
    заменить его на общее количество значило бы внести в документацию
    новую ошибку.
    """
    monkeypatch.setattr(check_docs, "count_tests", lambda: 321)
    write(
        project, "doc/presentation.md",
        "| `pytest tests -q` | 306 тест |\n"
        "| `test_collector.py` | 18 тестов |\n"
        "исправлено и закрыто 14 тестами\n",
    )

    check_docs.fix_totals()

    text = (project / "doc" / "presentation.md").read_text(encoding="utf-8")
    assert "18 тестов" in text
    assert "14 тестами" in text
    assert "321 тест" in text


def test_fix_is_idempotent(project, monkeypatch):
    """Повторный запуск ничего не меняет.

    Иначе каждый прогон CI, вызывающий `--fix`, трогал бы файлы и
    создавал бы лишние изменения в рабочей копии.
    """
    monkeypatch.setattr(check_docs, "count_tests", lambda: 321)
    write(project, "README.md", "| `pytest tests -q` | 306 тест |\n")

    first = check_docs.fix_totals()
    before = (project / "README.md").read_text(encoding="utf-8")
    second = check_docs.fix_totals()

    assert len(first) == 1
    assert second == []
    assert (project / "README.md").read_text(encoding="utf-8") == before


def test_fix_does_not_touch_plan(project, monkeypatch):
    """`PLAN.md` не правится: он фиксирует прошлые состояния.

    Подмена числа в исторической записи уничтожила бы именно то,
    ради чего она написана.
    """
    monkeypatch.setattr(check_docs, "count_tests", lambda: 321)
    (project / "PLAN.md").write_text(
        "тогда было 205 тестов\n", encoding="utf-8"
    )

    check_docs.fix_totals()

    assert "205 тестов" in (project / "PLAN.md").read_text(encoding="utf-8")


def test_unknown_key_is_rejected(project):
    """Неизвестный ключ — ошибка вызова, а не молчаливое игнорирование."""
    assert check_docs.main(["--что-то"]) == 2


def test_main_reports_success(project, capsys):
    """Успешный прогон говорит, что именно проверено."""
    write(project, "README.md", "| `pytest tests -q` | 306 тест |\n")
    assert check_docs.main([]) == 0
    assert "306" in capsys.readouterr().out


def test_matching_number_passes(project):
    """Число, совпадающее с фактическим, замечанием не считается."""
    write(project, "README.md", "| `pytest tests -q` | 306 тест |")
    assert check_docs.check_test_count() == []


def test_stale_number_points_at_the_place(project):
    """Замечание называет файл и строку, а не требует искать по проекту.

    Сообщение «в тексте 302, в tests/ — 306» не говорит, где именно
    неверное число: в шести файлах их десятки. Указание места сокращает
    правку до одной строки.
    """
    write(
        project, "README.md",
        "первая строка\n"
        "| `pytest tests -q` | 302 тестов |\n"
        "третья строка\n",
    )
    problems = check_docs.check_test_count()

    assert len(problems) == 1
    assert "302 тестов" in problems[0]
    assert "306" in problems[0]
    assert "README.md:2" in problems[0]


def test_all_stale_places_are_named(project):
    """Указаны все места, а не только первое найденное.

    Иначе пришлось бы запускать проверку несколько раз, по одному
    исправлению за запуск.
    """
    write(project, "README.md", "| `pytest tests -q` | 302 тестов |")
    write(project, "AGENTS.md", "- `tests/` — 302 тестов, валидатор\n")
    write(project, "doc/github_actions.md", "| Тесты | `tests/` (302 тестов) |\n")

    problems = check_docs.check_test_count()

    assert len(problems) == 1
    assert "README.md:1" in problems[0]
    assert "AGENTS.md:1" in problems[0]
    assert "github_actions.md:1" in problems[0]


def test_absent_total_is_reported(project):
    """Если итог нигде не назван — это тоже повод спросить.

    Молча пропущенная проверка выглядит как успешная.
    """
    problems = check_docs.check_test_count()
    assert len(problems) == 1
    assert "не найдено" in problems[0]


def test_per_file_counts_are_not_mistaken_for_the_total(project):
    """Число тестов **одного файла** — не итоговое.

    Два вида, оба встречаются в документации: инструментальный падеж
    («закрыто 14 тестами») и число рядом с именем файла в кавычках.
    Первое отличается формой слова, второе — соседством с путём.
    """
    write(
        project, "doc/presentation.md",
        "| `pytest tests -q` | 306 тест |\n"
        "| `test_collector.py` | 18 тестов |\n"
        "исправлено и закрыто 14 тестами\n",
    )
    assert check_docs.check_test_count() == []


def test_a_total_in_an_unusual_wording_is_still_caught(project):
    """Формулировка значения не имеет значения.

    Первая версия проверяла по белому списку шаблонов, и из восьми
    реальных упоминаний итогового числа узнавала пять: устаревшее
    число в трёх местах проходило молча. Теперь берутся все вхождения,
    и это ловит в том числе формулировки, о которых автор не думал.
    """
    for name, text in (
        ("AGENTS.md", "- `tests/` — 302 тестов, валидатор\n"),
        ("doc/presentation.md", "- три job. 302 тестов, полный прогон\n"),
        ("doc/github_actions.md", "самого workflow и 302 тестов. Дальше.\n"),
    ):
        write(project, name, text)

    problems = check_docs.check_test_count()

    assert len(problems) == 1
    for place in ("AGENTS.md:1", "presentation.md:1", "github_actions.md:1"):
        assert place in problems[0]


def test_historical_plan_is_not_held_to_the_current_total(project, tmp_path):
    """`PLAN.md` не входит в проверку: он фиксирует прошлые состояния.

    Сверять его итоги с сегодняшним числом тестов — значит требовать
    от истории соответствовать настоящему.
    """
    assert "PLAN.md" not in check_docs.CURRENT_DOCS


# ---------------------------------------------------------------------------
# Артефакты прогона
# ---------------------------------------------------------------------------


def test_missing_artifacts_are_not_an_error(project):
    """Свежий клон без артефактов — обычное состояние, а не расхождение.

    Job `test` в CI идёт до обучения, и `data/metadata` там ещё нет.
    Если бы проверка на этом падала, её пришлось бы отключать, а вместе
    с ней и всё, ради чего она сделана.
    """
    assert check_docs.check_manifest_facts("") == []


def test_empty_metadata_directory_is_not_an_error(project):
    """Каталог создан, но батчей ещё нет — тоже не расхождение."""
    (project / "data" / "metadata").mkdir(parents=True)
    assert check_docs.check_manifest_facts("") == []


def test_partial_run_is_reported(project):
    """Прогон, остановившийся на середине, виден как неполный.

    Здесь молчать нельзя: документация описывает результат 48 батчей,
    а на диске их 20, и читатель примет одно за другое.
    """
    metadata = project / "data" / "metadata"
    metadata.mkdir(parents=True)
    for index in range(20):
        (metadata / f"run_manifest_{index:04d}.json").write_text(
            json.dumps({"batch_idx": index}), encoding="utf-8"
        )

    problems = check_docs.check_manifest_facts("")

    assert any("20 из 20" in problem for problem in problems)


def test_unreadable_manifest_is_named_but_does_not_stop_the_rest(project):
    """Битый манифест называется и не отменяет разбор остальных.

    Молча пропустить его нельзя — тогда история прогона выглядит
    полной, будучи на один батч короче. Но и обрывать проверку тоже
    нельзя: остальные 47 батчей ещё сверяемы.
    """
    metadata = project / "data" / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "run_manifest_0000.json").write_text("{не json", encoding="utf-8")
    for index in range(1, 48):
        (metadata / f"run_manifest_{index:04d}.json").write_text(
            json.dumps({
                "batch_idx": index,
                "preprocessor": {"n_features": 306},
            }),
            encoding="utf-8",
        )

    problems = check_docs.check_manifest_facts("")

    assert any("run_manifest_0000.json" in p for p in problems)
    assert any("47 из 48" in p for p in problems)
    # Число признаков по оставшимся батчам всё же сверено.
    assert not any("признак" in p for p in problems)


def test_feature_count_mismatch_is_reported(project):
    """Число признаков сверяется с манифестами."""
    metadata = project / "data" / "metadata"
    metadata.mkdir(parents=True)
    for index in range(48):
        (metadata / f"run_manifest_{index:04d}.json").write_text(
            json.dumps({
                "batch_idx": index,
                "preprocessor": {"n_features": 412},
            }),
            encoding="utf-8",
        )

    problems = check_docs.check_manifest_facts("306 признаков после предобработки")

    assert any("412" in problem for problem in problems)


# ---------------------------------------------------------------------------
# Ссылки на файлы
# ---------------------------------------------------------------------------


def test_missing_referenced_file_is_reported(project):
    """Упомянутый в документации путь должен существовать.

    Битая ссылка хуже отсутствующей: читатель идёт по ней и упирается
    в 404, считая это ошибкой проекта.
    """
    problems = check_docs.check_referenced_files("см. `src/nonexistent.py`")
    assert any("src/nonexistent.py" in problem for problem in problems)


def test_existing_reference_passes(project):
    """Существующий путь замечанием не считается."""
    assert check_docs.check_referenced_files(
        "см. `src/`, `run.py`, `config.yaml` и `doc/task.md`"
    ) == []


def test_reference_without_backticks_is_ignored(project):
    """Проверяются только ссылки в кавычках.

    Пути в обычном тексте упомянуты для человека, и требовать от них
    формата значит запрещать нормальный прозвый текст.
    """
    assert check_docs.check_referenced_files("см. src/nonexistent.py") == []
