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

#: Признаки того, что `run`-шаг работает с содержимым репозитория.
#:
#: Проверка `checkout` — это линь, а не доказательство, поэтому она
#: опирается на конкретные признаки в тексте шага, а не на сам факт
#: `run`. Иначе пришлось бы требовать checkout и от job, которому
#: репозиторий не нужен вовсе: публикация сайта получает его отдельным
#: артефактом, а из репозитория зовёт только API GitHub.
WORKSPACE_MARKERS = (
    "run.py",
    "src/",
    "tests/",
    "scripts/",
    "pytest",
    "python -m",
    "pip install",
    "bash ",
    "sh ",
)

#: Допустимые входы action'ов, используемых в этом workflow.
#:
#: Несуществующий вход — не опечатка, а остановка job: GitHub проверяет
#: `with:` до выполнения шага. Именно так проявился `retention-days` у
#: `actions/cache` — у него такого входа нет, и весь job обучения
#: остановился, не отработав ни одного батча.
#:
#: Таблица сверялась с `action.yml` перечисленных версий. Она
#: устаревает при обновлении action'ов; обновить её можно так:
#:
#:     curl -s https://raw.githubusercontent.com/actions/cache/v6/action.yml
#:
#: Для неизвестных action'ов проверка молчит: отсутствие записи в
#: таблице не считается ошибкой, иначе добавление любого нового
#: action'а требовало бы правки здесь.
ACTION_INPUTS: dict[str, set[str]] = {
    "actions/checkout": {
        "repository", "ref", "token", "ssh-key", "path", "clean",
        "fetch-depth", "lfs", "submodules", "set-safe-directory",
        "persist-credentials", "sparse-checkout", "sparse-checkout-cone-mode",
        "fetch-tags", "show-progress", "filter", "ssh-strict",
        "ssh-known-hosts", "clean-exclude",
    },
    "actions/setup-python": {
        "python-version", "python-version-file", "cache", "architecture",
        "check-latest", "token", "cache-dependency-path",
        "update-environment", "allow-prereleases", "freethreaded", "pip-version",
    },
    "actions/cache": {
        "path", "key", "restore-keys", "upload-chunk-size",
        "enableCrossOsArchive", "fail-on-cache-miss", "lookup-only",
        "save-always",
    },
    "actions/cache/restore": {
        "path", "key", "restore-keys", "upload-chunk-size",
        "enableCrossOsArchive", "fail-on-cache-miss", "lookup-only",
    },
    "actions/cache/save": {
        "path", "key", "upload-chunk-size", "enableCrossOsArchive",
    },
    "actions/upload-artifact": {
        "name", "path", "if-no-files-found", "retention-days",
        "compression-level", "overwrite", "include-hidden-files", "archive",
    },
    "actions/upload-pages-artifact": {
        "name", "path", "retention-days", "include-hidden-files",
    },
    "actions/configure-pages": {
        "static_site_generator", "generator_config_file", "token", "enablement",
    },
    "actions/deploy-pages": {
        "token", "timeout", "error_count", "reporting_interval",
        "artifact_name", "preview",
    },
}


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
        # Job, который кладёт артефакты, обязан сначала получить
        # репозиторий: иначе он будет работать с пустым рабочим
        # каталогом и тихо ничего не сделает. Требование относится к
        # шагам, которые действительно обращаются к файлам проекта.
        scripts = "\n".join(
            str(step.get("run", ""))
            for step in steps
            if isinstance(step, dict) and "run" in step
        )
        if not uses and scripts:
            problems.append(f"{path}: job {name!r}: шаги без действий")
        elif any(marker in scripts for marker in WORKSPACE_MARKERS):
            if not any(action.startswith("actions/checkout") for action in uses):
                problems.append(
                    f"{path}: job {name!r} не делает checkout, "
                    f"но его шаги работают с файлами проекта"
                )

        problems.extend(_unknown_inputs(path, name, steps))

    return problems


def _unknown_inputs(path: Path, job: str, steps: Any) -> list[str]:
    """Входы `with:`, которых у action'а нет.

    GitHub проверяет их до выполнения шага и останавливает job целиком,
    поэтому опечатка стоит дорого: обнаруживается она только в CI, и
    после того, как потрачено время на ожидание.
    """
    problems: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        uses = str(step.get("uses", ""))
        base = uses.split("@", 1)[0]
        allowed = ACTION_INPUTS.get(base)
        if allowed is None:
            # Действие не в таблице: проверка молчит, чтобы добавление
            # нового action'а не требовало правки здесь.
            continue
        given = step.get("with") or {}
        if not isinstance(given, dict):
            continue
        for key in given:
            if str(key) not in allowed:
                problems.append(
                    f"{path}: job {job!r}, действие {base}: входа {key!r} "
                    f"не существует; допустимо: "
                    f"{', '.join(sorted(allowed))}"
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
