"""Сбор и эмуляция потока данных.

Исходная реализация держала пути в значениях по умолчанию
(`output_dir='data/raw_batches'`, `state.json` в текущем каталоге) и
записывала в состояние **относительные** пути к батчам, из-за чего
конвейер зависел от текущего рабочего каталога (D-13, D-20). Здесь все
пути берутся из конфигурации и разрешаются относительно корня проекта.
"""

from __future__ import annotations

import hashlib
import logging
import zipfile
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.config import Config
from src.state import StateStore
from src.utils import parse_datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Источник данных
# ---------------------------------------------------------------------------


def _file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """Хэш файла источника — попадает в состояние и манифест (АР-6)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()[:16]


def extract_source(config: Config, force: bool = False) -> Path:
    """Подготовить исходный CSV: распаковать архив, если он ещё не распакован.

    Returns:
        Путь к CSV с данными.
    """
    csv_path = config.path("source_csv")
    zip_path = config.path("source_zip")

    if csv_path.is_file() and not force:
        return csv_path

    if not zip_path.is_file():
        raise FileNotFoundError(
            f"Источник данных не найден: ни {csv_path.name}, ни {zip_path.name}. "
            f"Положите архив в {zip_path.parent}."
        )

    logger.info("Распаковка %s", zip_path.name)
    target_dir = csv_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise RuntimeError(f"В архиве {zip_path.name} нет CSV-файлов")
        if len(members) > 1:
            logger.warning(
                "В архиве %d CSV-файлов, используется первый: %s", len(members), members[0]
            )
        with archive.open(members[0]) as source, open(csv_path, "wb") as target:
            target.write(source.read())

    return csv_path


def dataset_hash(config: Config) -> str | None:
    """Хэш исходных данных, если источник доступен."""
    csv_path = config.path("source_csv")
    if csv_path.is_file():
        return _file_hash(csv_path)
    zip_path = config.path("source_zip")
    return _file_hash(zip_path) if zip_path.is_file() else None


# ---------------------------------------------------------------------------
# Разбиение на батчи
# ---------------------------------------------------------------------------


def split_into_batches(
    config: Config,
    source_path: Path,
    store: StateStore,
    reset_progress: bool = True,
) -> list[str]:
    """Разбить исходный набор на батчи по временной колонке.

    Батчи именуются по периоду `batch_YYYY-MM.csv` и содержат все колонки
    исходного набора: очистка применяется позже, на этапе `update`, чтобы
    хранилище сырых данных оставалось сырым.

    Args:
        config: конфигурация.
        source_path: распакованный исходный CSV.
        store: состояние прогона, в которое попадёт список батчей.
        reset_progress: сбрасывать ли отметку о пройденных батчах, если
            список батчей не изменился. По умолчанию — да, но
            конвейер передаёт `False`: повторный `init` на тех же данных
            не должен откатывать уже накопленное обучение, иначе
            инкрементальный сценарий CI работает как обучение с нуля.

    Returns:
        Список относительных путей к батчам в хронологическом порядке.
    """
    time_column = config.time_column
    date_format = config["batching"].get("date_format")
    period = config["batching"].get("size", "month")

    if not _resolve_time_period(period):
        raise ValueError(
            f"batching.size={period!r} не поддерживается; допустимо: month, quarter, year"
        )

    logger.info("Чтение источника %s", source_path.name)
    frame = pd.read_csv(
        source_path,
        parse_dates=[time_column],
        date_format=date_format,
        low_memory=False,
    )

    if time_column not in frame.columns:
        raise KeyError(f"В источнике нет временной колонки {time_column!r}")

    unparsed = int(frame[time_column].isna().sum())
    if unparsed:
        logger.warning(
            "Не удалось разобрать %d значений в колонке %s", unparsed, time_column
        )

    frame = frame.dropna(subset=[time_column]).sort_values(time_column).reset_index(drop=True)
    frame["__period__"] = getattr(frame[time_column].dt, f"to_period")(
        "M" if period == "month" else "Q" if period == "quarter" else "Y"
    )

    batches_dir = config.batches_dir
    batches_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for value, group in frame.groupby("__period__", sort=True):
        payload = group.drop(columns=["__period__"])
        path = batches_dir / f"batch_{value}.csv"
        payload.to_csv(path, index=False)
        written.append(str(path.relative_to(config.root)))
        logger.info("Батч %s: %d строк", value, len(payload))

    store.init_batches(
        written,
        dataset_hash=dataset_hash(config),
        reset=reset_progress,
    )
    logger.info("Создано батчей: %d (период: %s)", len(written), period)
    return written


def _resolve_time_period(period: str) -> bool:
    return period in ("month", "quarter", "year")


# ---------------------------------------------------------------------------
# Чтение батчей
# ---------------------------------------------------------------------------


def read_batch(path: Path, config: Config) -> pd.DataFrame:
    """Прочитать батч, разрешив путь относительно корня проекта.

    `low_memory=False` задан намеренно: при разборке по блокам часть
    повреждённых значений молча превращается в `NaN` (в этом наборе так
    ведёт себя `EFFECTIVE_YR`, см. D-26), и проверка типов не смогла бы
    увидеть исходный мусор.

    Временная колонка приводится к датам сразу: в батчах она записана в
    ISO, тогда как формат конфигурации описывает исходный набор, поэтому
    используется tolerant-разбор с перебором форматов.
    """
    frame = pd.read_csv(path, low_memory=False)

    expected = config.get("schema", {}).get("required_columns")
    if expected:
        missing = [column for column in expected if column not in frame.columns]
        if missing:
            raise KeyError(
                f"В батче {path.name} отсутствуют обязательные колонки: {missing}"
            )

    time_column = config.time_column
    if time_column in frame.columns:
        frame[time_column] = parse_datetime(
            frame[time_column], config["batching"].get("date_format")
        )

    return frame


def iter_batches(
    config: Config, store: StateStore, limit: int | None = None
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Перебирать батчи начиная с первого необработанного."""
    start = store.last_processed + 1
    total = len(store)
    end = total if limit is None else min(start + limit, total)
    for index in range(start, end):
        yield index, read_batch(config.root / store.batches[index], config)


def batch_path(config: Config, store: StateStore, index: int) -> Path:
    """Абсолютный путь к батчу по индексу."""
    return config.root / store.batches[index]


def batch_label(store: StateStore, index: int) -> str:
    """Человекочитаемое имя батча, например `batch_2014-07`."""
    return Path(store.batches[index]).stem
