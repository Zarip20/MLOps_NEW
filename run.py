"""Точка входа конвейера.

Скрипт только разбирает аргументы, настраивает логирование и вызывает
оркестратор. Вся логика этапов находится в `src/pipeline.py` (АР-3).

Коды выхода:
    0 — успех;
    1 — ошибка конфигурации или непредвиденный сбой;
    2 — ошибка аргументов (argparse).

Требование §4 задания 1: `-mode update` возвращает bool успеха. Здесь этот
bool превращается в код возврата процесса, чтобы CI мог на него опереться
(АР-11): раньше `update()` печатал константу `True` и всегда завершался
нулевым кодом, а workflow без `set -e` это не замечал.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import ConfigError, load_config
from src.logging_setup import configure_logging
from src.pipeline import Pipeline, PipelineError

EXIT_OK = 0
EXIT_ERROR = 1

LOG = logging.getLogger("mlops")


def build_parser() -> argparse.ArgumentParser:
    """Описание командной строки."""
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="MLOps-конвейер обработки потоковых табличных данных",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python run.py -mode init\n"
            "  python run.py -mode init -reset       # обучение с нуля\n"
            "  python run.py -mode update -n 12\n"
            "  python run.py -mode sync\n"
            "  python run.py -mode inference -file data/new_batch.csv\n"
            "  python run.py -mode summary\n"
            "  python run.py -mode publish\n"
        ),
    )
    parser.add_argument(
        "-mode",
        required=True,
        choices=["init", "sync", "update", "inference", "summary", "publish"],
        help="Режим работы конвейера",
    )
    parser.add_argument(
        "-n",
        "--n-batches",
        type=int,
        default=None,
        help="Сколько батчей обработать (по умолчанию — все оставшиеся)",
    )
    parser.add_argument(
        "-file",
        dest="file_path",
        help="Путь к файлу с данными для инференса",
    )
    parser.add_argument(
        "-config",
        dest="config_path",
        default=None,
        help="Путь к конфигурационному файлу (по умолчанию — config.yaml в корне)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Уровень логирования (перекрывает значение из конфигурации)",
    )
    parser.add_argument(
        "-reset",
        "--reset",
        dest="reset",
        action="store_true",
        help=(
            "С режимом init: начать с нуля, сбросив уже накопленное "
            "обучение, даже если данные не изменились"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Основной цикл. Возвращает код выхода процесса."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config_path)
    except ConfigError as error:
        # Логирование ещё не настроено — пишем в поток ошибок напрямую.
        print(f"Ошибка конфигурации:\n{error}", file=sys.stderr)
        return EXIT_ERROR

    log_level = args.log_level or config.run.get("log_level", "INFO")
    configure_logging(config, level=log_level)

    if args.mode == "inference" and not args.file_path:
        parser.error("Режим inference требует параметра -file")

    try:
        pipeline = Pipeline(config)
    except Exception as error:  # noqa: BLE001
        LOG.exception("Не удалось инициализировать конвейер: %s", error)
        return EXIT_ERROR

    try:
        return _dispatch(pipeline, args)
    except PipelineError as error:
        LOG.error("%s", error)
        return EXIT_ERROR
    except KeyboardInterrupt:
        LOG.warning("Остановлено пользователем. Состояние сохранено.")
        return EXIT_ERROR
    except Exception as error:  # noqa: BLE001
        LOG.exception("Непредвиденная ошибка: %s", error)
        return EXIT_ERROR


def _dispatch(pipeline: Pipeline, args: argparse.Namespace) -> int:
    """Выполнить запрошенный режим."""
    config = pipeline.config
    mode = args.mode

    if mode == "init":
        info = pipeline.init(reset=args.reset)
        if info["reused"]:
            print(
                f"Набор данных не изменился: батчей {info['batches']}, "
                f"уже обработано {info['processed']} (прогресс сохранён)"
            )
        else:
            print(f"Инициализация завершена: батчей {info['batches']}")
        print(f"  первый: {info['first']}")
        print(f"  последний: {info['last']}")
        print(f"  хэш данных: {info['dataset_hash']}")
        print(f"  сборщик данных: models/{info['collector']}")
        return EXIT_OK

    if mode == "sync":
        added = pipeline.sync_batches()
        if added:
            print(f"Добавлено батчей в состояние: {len(added)}")
            for name in added:
                print(f"  {name}")
        else:
            print("Новых батчей в каталоге не найдено")
        return EXIT_OK

    if mode == "update":
        limit = args.n_batches
        if limit is None:
            limit = config.pipeline.get("default_batches")
        processed = pipeline.update(limit=limit)
        # False, если батчи закончились раньше запрошенного объёма либо
        # обработка прервалась сбоем (АР-11).
        if processed == 0 and pipeline.state.has_next:
            print("Обработка не выполнена", file=sys.stderr)
            return EXIT_ERROR
        print(f"Обработано батчей: {processed}")
        return EXIT_OK

    if mode == "inference":
        output = pipeline.inference(Path(args.file_path))
        print(f"Результат сохранён: {output}")
        return EXIT_OK

    if mode == "publish":
        # Сайт для GitHub Pages (задание 2, пункт 2.b.iii). Сборкой
        # занимается слой представлений, точка входа только запрашивает
        # результат: иначе вывод стал бы частью контроллера (7.b.iv).
        site = pipeline.publish()
        print(f"Сайт собран: {site}")
        return EXIT_OK

    if mode == "summary":
        report, dashboard = pipeline.summary()
        print(f"Отчёт сохранён:     {report}")
        print(f"Дашборд сохранён:  {dashboard}")
        return EXIT_OK

    parser_error = f"Неизвестный режим: {mode}"
    print(parser_error, file=sys.stderr)
    return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
