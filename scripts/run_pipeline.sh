#!/usr/bin/env bash
# Единая точка запуска конвейера.
#
# Скрипт сознательно один и для локального запуска, и для CI (АР-15):
# расхождение «локально работает, в CI иначе» — самая частая причина
# потери баллов за «успешное обучение модели». Проверка кодов возврата
# обязательна: без `set -e` упавший батч не остановил бы job, и тот
# остался бы зелёным.
set -euo pipefail

N_BATCHES="${1:-}"
PYTHON="${PYTHON:-python}"

log() {
    printf '=== %s ===\n' "$1"
}

log "Окружение"
"$PYTHON" --version
"$PYTHON" -c "import pandas, numpy, sklearn, mlxtend, yaml; \
print('pandas', pandas.__version__); \
print('numpy', numpy.__version__); \
print('scikit-learn', sklearn.__version__); \
print('mlxtend', mlxtend.__version__)"

log "Инициализация батчей"
"$PYTHON" run.py -mode init

if [ -n "$N_BATCHES" ]; then
    log "Обновление модели: $N_BATCHES батчей"
    "$PYTHON" run.py -mode update -n "$N_BATCHES"
else
    log "Обновление модели: все оставшиеся батчи"
    "$PYTHON" run.py -mode update
fi

log "Формирование отчёта"
"$PYTHON" run.py -mode summary

log "Сборка сайта с дашбордом истории обучения"
# Каталог сайта очищается перед сборкой, поэтому в него не попадают
# артефакты прошлых прогонов. Публикация не должна ронять успешное
# обучение, поэтому при сбое здесь ставится предупреждение, а не ошибка.
if ! "$PYTHON" run.py -mode publish; then
    echo "Предупреждение: сайт не собран, обучение при этом успешно" >&2
fi

log "Описание сборщика данных"
if ! "$PYTHON" -m src.collector --describe; then
    echo "Предупреждение: не удалось прочитать каталог батчей" >&2
fi

log "Проверка инференса на последнем батче"
INFERENCE_INPUT="data/raw_batches/$(ls -1 data/raw_batches | sort | tail -n 1)"
if [ -f "$INFERENCE_INPUT" ]; then
    "$PYTHON" run.py -mode inference -file "$INFERENCE_INPUT"
else
    echo "Предупреждение: нет батчей для проверки инференса" >&2
fi

log "Готово"
echo "Состояние: $(ls -1 data/metadata 2>/dev/null | wc -l) файлов метаданных"
echo "Модели:    $(ls -1 models 2>/dev/null | tr '\n' ' ')"
echo "Отчёты:    $(ls -1 reports 2>/dev/null | tr '\n' ' ')"
echo "Сайт:      $(ls -1 site 2>/dev/null | tr '\n' ' ')"
