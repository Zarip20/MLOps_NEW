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

    python -m tests.check_docs
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

#: Формулировки, которыми в разных файлах называется ВСЕГО тестов.
#: По одной формулировке ищутся числа, по другой — не ищется ничего:
#: «в файле test_collector.py 18 тестов» это не итог, и сравнивать его
#: с общим числом бессмысленно.
TOTAL_PATTERNS = (
    r"(\d{2,4})\s+тест(?:а|ов)?\s*\|",              # README: строка таблицы
    r"#\s*(\d{2,4})\s+тест",                          # github_actions: комментарий
    r"tests/\s*\((\d{2,4})\s+тест",                   # github_actions: таблица
    r"`pytest tests -q`\s*\|\s*(\d{2,4})\s+тест",      # README: команда
    r"тестов:\s*(\d{2,4})",                           # grade: раздел проверки
    r"(\d{2,4})\s+тест(?:а|ов)?[,\.]?\s*$",            # хвост строки
)


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


def check_test_count(text: str) -> list[str]:
    """Все названные итоговые числа должны совпадать между собой."""
    found = {
        int(value)
        for pattern in TOTAL_PATTERNS
        for value in re.findall(pattern, text)
    }
    actual = count_tests()
    if not found:
        return ["в документации не найдено ни одного итогового числа тестов"]
    return [
        f"итоговое число тестов: в тексте {sorted(found)}, в tests/ — {actual}"
    ] if found != {actual} else []


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
    for path in sorted(metadata.glob("run_manifest_*.json")):
        try:
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    if not manifests:
        return []

    problems: list[str] = []
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
            f"манифестов {processed}, а документация описывает прогон 48 батчей"
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
        check_test_count(text)
        + check_referenced_files(text)
        + check_manifest_facts(text)
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    if arguments:
        print("Проверка принимает пустой список аргументов или ни одного")
        return 2

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
