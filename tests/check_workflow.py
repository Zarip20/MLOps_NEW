"""Проверка синтаксиса workflow-файлов GitHub Actions.

Опечатка в YAML не даёт запуска, а сообщение об ошибке указывает на
номер строки в самом workflow, а не на смысл проблемы. Здесь файл
разбирается и проверяется на те ошибки, которые чаще всего остаются
незамеченными: несуществующий job в `needs`, отсутствующий checkout,
дублирующиеся ключи и неверная структура шагов.

Запуск:

    python -m tests.check_workflow .github/workflows/main.yml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

#: Шаги, без которых job, ставящий `needs` на обучение, бессмыслен.
REQUIRED_ACTIONS = ("actions/checkout", "actions/setup-python")


def load(path: Path) -> dict[str, Any]:
    """Разобрать workflow. Ключ `on` YAML превращает в True — учитываем это."""
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: ожидался словарь, получено {type(data).__name__}")
    return data


def triggers(data: dict[str, Any]) -> list[str]:
    return list(data.get("on") or data.get(True) or [])


def check(path: Path) -> list[str]:
    """Вернуть список проблем; пустой список — файл в порядке."""
    data = load(path)
    problems: list[str] = []

    jobs = data.get("jobs") or {}
    if not jobs:
        return [f"{path}: нет ни одного job"]

    if not triggers(data):
        problems.append(f"{path}: не заданы триггеры (on)")

    for name, job in jobs.items():
        needs = job.get("needs")
        needed = [needs] if isinstance(needs, str) else list(needs or [])
        for dependency in needed:
            if dependency not in jobs:
                problems.append(f"{path}: job {name!r} ждёт несуществующий {dependency!r}")

        steps = job.get("steps") or []
        if not steps:
            problems.append(f"{path}: job {name!r} не содержит шагов")

        # Проверяется каждый шаг по отдельности: шаг с одним лишь
        # `name` — это опечатка, и GitHub выполняет его молча, ничего
        # не делая, а job остаётся зелёным.
        for step in steps:
            if not isinstance(step, dict):
                problems.append(f"{path}: job {name!r}: шаг не является объектом")
                continue
            if "uses" not in step and "run" not in step:
                label = step.get("name", "без имени")
                problems.append(
                    f"{path}: job {name!r}: шаг {label!r} без действий "
                    f"(нет ни uses, ни run)"
                )

        uses = [
            str(step.get("uses", ""))
            for step in steps
            if isinstance(step, dict)
        ]
        runs_code = any(
            isinstance(step, dict) and "run" in step for step in steps
        )

        # Job, который кладёт артефакты, обязан сначала получить репозиторий
        # и поставить интерпретатор: иначе он будет работать с пустым
        # рабочим каталогом.
        if any(action.startswith(REQUIRED_ACTIONS) for action in uses) or runs_code:
            if not any(action.startswith("actions/checkout") for action in uses):
                problems.append(
                    f"{path}: job {name!r} не делает checkout, "
                    f"но использует код или готовое окружение"
                )

    return problems


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    if not arguments:
        arguments = [".github/workflows/main.yml"]

    failures = 0
    for item in arguments:
        path = Path(item)
        if not path.is_file():
            print(f"{path}: файл не найден")
            failures += 1
            continue
        try:
            problems = check(path)
        except (ValueError, yaml.YAMLError) as error:
            print(f"{path}: не разобран — {error}")
            failures += 1
            continue

        if problems:
            failures += 1
            for problem in problems:
                print(f"ОШИБКА {problem}")
        else:
            data = load(path)
            print(
                f"{path}: в порядке "
                f"(job: {', '.join(data['jobs'])}; триггеры: {', '.join(triggers(data))})"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
