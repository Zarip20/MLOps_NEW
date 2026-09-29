# AGENTS.md

## Проект
MLOps-система обработки потоковых табличных данных (учебный проект).
Требования: `doc/Masters_MLOps_task_1.pdf`, `doc/MLOps_task_2.pdf`.

## Правила работы
1. Работай ТОЛЬКО над задачей из текущего промпта. Не трогай остальное.
2. Перед изменениями дай план и diff. Не применяй без подтверждения.
3. Перед ответом перечитай AGENTS.md и PLAN.md.
4. Не пиши код, если просят только план.

## Команды
- Запуск: `.\.venv\Scripts\python run.py -mode <init|sync|update|inference|summary|publish>`
- Тесты: `.\.venv\Scripts\python -m pytest tests -q`
- Проверка workflow: `.\.venv\Scripts\python -m tests.check_workflow .github/workflows/main.yml`
- Сборщик данных: `.\.venv\Scripts\python -m src.collector --describe|--append <файл>`
- Сборка сайта: `.\.venv\Scripts\python -m src.publish --out site`
- CI: `bash scripts/run_pipeline.sh <N>` — тот же скрипт, что зовёт workflow

## Структура
- `run.py` — только разбор аргументов; вся логика в `src/pipeline.py`
- `config.yaml` — единственный источник настроек, проверяется при старте
- `src/` — модули конвейера (см. дерево в `README.md`)
- `tests/` — 203 теста, `check_workflow.py` — валидатор workflow
- `.github/workflows/main.yml` — job `test → train → publish`
- `doc/` — требования, оценка, `github_actions.md` (инструкция по CI)
- `PLAN.md` — авторитетная запись хода работы и решений Р-1…Р-6

## Правила, выведенные из практики
- Не редактировать файлы через PowerShell `Get-Content`/`Set-Content`:
  ломает кириллицу. Только инструменты правки файлов.
- `Config` неизменяемый (frozen dataclass) — в тестах собирается новый.
- `git` в системе нет; первый `push` делает пользователь.
