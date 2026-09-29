"""Настройка логирования (АР-12).

Единая точка конфигурации логов. Прежняя реализация настраивала вывод
в двух разных файлах (`run.py` и мёртвом `src/run.py` давали разное
поведение), а `README` и `task.md` утверждали, что сообщения выводятся
в консоль, хотя фактически писались только в файл.

Теперь:
* сообщения идут одновременно в консоль и в `training.log`;
* в лог пишутся только сообщения самих модулей (`mlops`, `src.*`),
  поэтому предупреждения pandas и sklearn не засоряют выгрузку;
* каждая запись содержит время, уровень и модуль-источник.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(config: Any, level: str = "INFO") -> logging.Logger:
    """Настроить корневой логгер и вернуть логгер приложения.

    Args:
        config: объект `Config`; используется путь к файлу журнала.
        level: уровень логирования.
    """
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        log_path = config.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as error:
        root.warning("Не удалось открыть файл журнала: %s", error)

    # Сторонние библиотеки — тише, иначе в выгрузке CI тонет в шуме.
    for noisy in ("matplotlib", "numexpr", "matplotlib.font_manager", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("mlops")
