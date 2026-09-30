"""Проверка документации: числа и ссылки должны сходиться с проектом.

Документация разошлась с кодом как минимум один раз: в `task.md` было
написано, что ассоциативные правила пересобираются на каждом батче,
хотя они фиксируются на первом. Написать это легко, обнаружить можно
только перечитыванием кода.

Здесь проверяются **только однозначные** утверждения — те, где число
стоит в одной и той же формулировке в нескольких файлах, плюс
существование всех упомянутых файлов. Проверка «похожих» чисел
сознательно не делается: она даёт столько ложных срабатываний, что
её перестают читать, — а ложное срабатывание хуже отсутствия проверки.

Запуск:

    python -m tests.check_docs           # только проверить
    python -m tests.check_docs --fix     # привести числа к фактическим

Ключ `--fix` нужен потому, что счётчик тестов меняется при каждом
новом тесте, а упоминаний восемь, и после них ещё и требуется
правильное склонение: «321 тест», но «322 теста». Вручную это каждый
раз приходилось бы вспоминать, и ошибка «321 тестов» повторялась.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import json

ROOT = Path(__file__).resolve().parent.parent

#: Документы, описывающие **текущее** состояние.
#:
#: `PLAN.md` сюда не входит сознательно: по замыслу это историческая
#: запись, и числа в нём верны для того момента, к которому относятся.
#: Сверять прошлые итоги с сегодняшним количеством тестов — значит
#: требовать от истории соответствовать настоящему.
CURRENT_DOCS = [
    "README.md", "AGENTS.md",
    "doc/task.md", "doc/grade.md", "doc/presentation.md",
    "doc/github_actions.md",
]

#: Любое упоминание количества тестов, а не только «красивые» формулировки.
#:
#: Первая версия проверяла по белому списку шаблонов («… тестов |»,
#: «# … тестов», «tests/ (… тестов)»). Список оказался дырявым: из
#: восьми реальных упоминаний итогового числа он узнавал пять, и
#: устаревшее число в трёх местах проходило молча — ровно то, ради чего
#: проверка и написана.
#:
#: Поэтому берутся **все** вхождения подряд, а исключаются два явно
#: отмеченных случая — см. `_is_per_file_count`.
COUNT_PATTERN = re.compile(r"\b(\d+)\s+тест\w*")


def _is_per_file_count(line: str, match: re.Match[str]) -> bool:
    """Упоминание количества тестов **одного файла**, а не всего набора.

    Различаются два случая, и оба встречаются в документации:

    * инструментальный падеж — «закрыто 14 тестами». Всегда про один
      файл: «закрыть 306 тестами» не имеет смысла, а «закрыто 306
      тестов» — имеет;
    * рядом стоит имя файла в кавычках — «| `test_meta.py` | 21
      тестов |».

    Всё остальное считается итогом. Ложное срабатывание неприятно, но
    оно заметно и исправляется за минуту, а ложное **пропускание**
    незаметно: устаревшее число годами выглядит как актуальное.
    """
    word = match.group(0).split(" ", 1)[1]
    if word.startswith("тестами") or word.startswith("тестах"):
        return True

    before = line[: match.start()]
    # Имя файла в кавычках, отделённое от числа только знаком таблицы.
    tail = before.rsplit("`", 2)
    if len(tail) >= 2 and re.search(r"\.py`?\s*\|?\s*$", tail[-2]):
        return True
    return False


def count_tests() -> int:
    """Число тестов.

    Сначала спрашивается у pytest — он учитывает параметризацию, а её
    используют около десятка проверок, и подсчёт по именам функций дал
    бы заниженное число. Если pytest недоступен, применяется разбор
    исходников с поправкой на число случаев параметризации.
    """
    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "--collect-only"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        result = None

    if result is not None and result.returncode == 0:
        found = re.findall(r"(\d+)\s+tests?\s+collected", result.stdout)
        if found:
            return int(found[-1])

    total = 0
    for path in (ROOT / "tests").glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                total += _parametrized_cases(node)
    return total


def _parametrized_cases(node: ast.FunctionDef) -> int:
    """Сколько тестов даёт функция: один либо по числу случаев."""
    for decorator in node.decorator_list:
        call = decorator if isinstance(decorator, ast.Call) else None
        target = decorator.func if call is not None else decorator
        name = getattr(target, "attr", getattr(target, "id", ""))
        if name != "parametrize":
            continue
        arguments = call.args if call is not None else []
        if len(arguments) < 2:
            continue
        cases = arguments[1]
        if isinstance(cases, (ast.List, ast.Tuple, ast.Set)):
            return max(len(cases.elts), 1)
    return 1


def docs_text() -> str:
    return "\n".join((ROOT / name).read_text(encoding="utf-8") for name in CURRENT_DOCS)


def _stated_counts() -> dict[int, list[str]]:
    """Итоговые числа тестов, где они упомянуты: число → список мест.

    Места возвращаются как `файл:строка`, потому что замечание «в тексте
    302, в тестах 306» требуло искать по всему проекту. С указанием
    строки правка занимает секунду.
    """
    found: dict[int, list[str]] = {}
    for name in CURRENT_DOCS:
        path = ROOT / name
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            for match in COUNT_PATTERN.finditer(line):
                if _is_per_file_count(line, match):
                    continue
                found.setdefault(int(match.group(1)), []).append(
                    f"{name}:{number}"
                )
    return found


def correct_form(number: int) -> str:
    """Правильная форма существительного после числительного.

    «321 тест», но «322 теста» и «325 тестов». Ошибка тут не в числах,
    а в грамматике, и она повторяется каждый раз при обновлении
    счётчика, — поэтому форма вычисляется, а не набирается вручную.
    """
    tail_hundreds, tail = number % 100, number % 10
    if 11 <= tail_hundreds <= 14:
        return "тестов"
    if tail == 1:
        return "тест"
    if 2 <= tail <= 4:
        return "теста"
    return "тестов"


def check_test_count() -> list[str]:
    """Все названные итоговые числа должны совпадать с фактическим."""
    found = _stated_counts()
    actual = count_tests()
    if not found:
        return ["в документации не найдено ни одного итогового числа тестов"]

    problems: list[str] = []
    for value in sorted(found):
        if value == actual:
            continue
        places = found[value]
        shown = ", ".join(places[:8])
        if len(places) > 8:
            shown += f" и ещё {len(places) - 8}"
        problems.append(
            f"указано «{value} тестов», а на самом деле {actual} "
            f"({actual} {correct_form(actual)}); исправить в {shown}"
        )
    return problems


def check_referenced_files(text: str) -> list[str]:
    """Все упомянутые в документации пути должны существовать."""
    problems: list[str] = []
    # `src/…`, `tests/…`, `scripts/…`, `doc/…` и корневые файлы —
    # с расширением или каталогом.
    pattern = r"`((?:src|tests|scripts|doc)/[\w./-]+|(?:run\.py|config\.yaml|README\.md|PLAN\.md|AGENTS\.md))`"
    for match in sorted(set(re.findall(pattern, text))):
        candidate = ROOT / match.rstrip(".,;)")
        if not candidate.exists():
            problems.append(f"документация ссылается на несуществующий путь: {match}")
    return problems


def check_manifest_facts(text: str) -> list[str]:
    """Числа, которые можно сверить с артефактами прогона."""
    metadata = ROOT / "data" / "metadata"
    if not metadata.is_dir():
        # Артефактов ещё нет — это не расхождение, а обычное состояние
        # свежего клона и job `test` в CI, который идёт до обучения.
        # Проверка обязана такие числа просто пропускать: иначе она
        # ругалась бы на то, чего ещё не существует, и её перестали бы
        # читать. О пропуске сказано в выводе.
        return []

    manifests: list[dict] = []
    unreadable: list[str] = []
    for path in sorted(metadata.glob("run_manifest_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            unreadable.append(f"{path.name} ({error})")
            continue
        if isinstance(payload, dict):
            manifests.append(payload)

    problems: list[str] = []
    if unreadable:
        # Битый манифест пропускается молча — из-за него одного
        # теряется весь разбор истории. Сообщение называет файлы.
        problems.append(
            f"не читаются манифесты: {', '.join(unreadable[:5])}"
        )

    if not manifests:
        return problems

    processed = len(manifests)
    if processed == 48:
        for match in sorted(set(re.findall(r"(\d+)\s+из\s+48", text))):
            if int(match) not in (47, 48):
                problems.append(
                    f"документация утверждает «{match} из 48 батчей», "
                    f"а обработано {processed}"
                )
    else:
        problems.append(
            f"прочитано манифестов {processed} из "
            f"{len(manifests) + len(unreadable)}, а документация описывает "
            f"прогон 48 батчей"
        )

    widths = {
        (manifest.get("preprocessor") or {}).get("n_features")
        for manifest in manifests
    }
    widths.discard(None)
    if len(widths) == 1:
        width = widths.pop()
        # 306 признаков после предобработки — единственное число такого
        # вида; «12 признаков» относится к инженерным, а не к итоговым.
        for match in sorted(set(re.findall(r"(\d{3})\s+признак", text))):
            if int(match) != width:
                problems.append(
                    f"признаков после предобработки: в тексте {match}, "
                    f"в манифестах {width}"
                )
    return problems


def check() -> list[str]:
    text = docs_text()
    return (
        check_test_count()
        + check_referenced_files(text)
        + check_manifest_facts(text)
    )


def fix_totals() -> list[str]:
    """Привести упоминания итогового числа к фактическому.

    Счётчик меняется при каждом новом тесте, а упоминаний восемь, и
    правильное склонение после них («321 тест», но «322 теста») при
   ходится ещё и вычислять. Оба действия выполняются здесь, потому
    что иначе ошибка повторяется каждый раз.

    Числа тестов **отдельных файлов** не трогаются: они не итоговые,
    и «18 тестов» про один файл — правильное утверждение.

    Returns:
        Список исправленных мест `файл:строка`.
    """
    actual = count_tests()
    replacement = f"{actual} {correct_form(actual)}"
    fixed: list[str] = []

    for name in CURRENT_DOCS:
        path = ROOT / name
        lines = path.read_text(encoding="utf-8").splitlines()
        changed = False

        for index, line in enumerate(lines):
            edits = [
                (match.start(), match.end(), replacement)
                for match in COUNT_PATTERN.finditer(line)
                if not _is_per_file_count(line, match)
            ]
            if not edits:
                continue
            for start, end, text in reversed(edits):
                line = line[:start] + text + line[end:]
            # Совпадение после замены означает, что число уже верное.
            # Без этой проверки `--fix` переписывал бы файлы при каждом
            # запуске, создавая изменения там, где их нет.
            if line == lines[index]:
                continue
            lines[index] = line
            fixed.append(f"{name}:{index + 1}")
            changed = True

        if changed:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return fixed


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])

    if any(item != "--fix" for item in arguments):
        print(
            "Проверка принимает ключ --fix (привести числа к фактическим) "
            "или не принимает ничего вовсе"
        )
        return 2

    if "--fix" in arguments:
        fixed = fix_totals()
        if not fixed:
            print("Итоговые числа в документации уже верны")
        else:
            print(f"Исправлено мест: {len(fixed)}")
            for place in fixed:
                print(f"  {place}")
            print("Проверьте повторным запуском без --fix")

    problems = check()
    metadata = ROOT / "data" / "metadata"
    artifacts = "сверено с артефактами" if (metadata / "run_manifest_0000.json").is_file() \
        else "артефактов прогона нет — числа сверены только с исходниками"

    if problems:
        for item in problems:
            print(f"ОШИБКА {item}")
        print(
            f"\nРасхождений: {len(problems)}. Обновите документацию "
            f"или перезапустите прогон."
        )
        return 1
    print(
        f"Документация сходится: тестов {count_tests()}, "
        f"упомянутые пути существуют, {artifacts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
