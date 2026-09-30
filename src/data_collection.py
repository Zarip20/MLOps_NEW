"""Сбор и эмуляция потока данных.

Исходная реализация держала пути в значениях по умолчанию
(`output_dir='data/raw_batches'`, `state.json` в текущем каталоге) и
записывала в состояние **относительные** пути к батчам, из-за чего
конвейер зависел от текущего рабочего каталога (D-13, D-20). Здесь все
пути берутся из конфигурации и разрешаются относительно корня проекта.

**Несколько источников (1.b.ii).** Раньше источник был один и зашит
в пару ключей `paths.source_zip` / `paths.source_csv`. Теперь их
список задаётся в конфигурации секцией `sources`, и каждый источник
приводится к одному виду через общий контракт `DataSource`: архив или
CSV, своё имя, свой префикс для имён батчей и своя доля в потоке.
Требование при этом не меняется — у всех источников одна и та же схема
колонок и одна целевая метка, иначе это были бы разные задачи.

Источники **складываются**: батчи нескольких файлов попадают в один
поток и сортируются по времени. Это позволяет, например, добавить
второй набор к первому, не заводя для него отдельный конвейер.
"""

from __future__ import annotations

import hashlib
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.config import Config
from src.state import StateStore
from src.utils import parse_datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Контракт источника данных
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSource:
    """Один источник данных, приведённый к общему виду.

    Контракт намеренно узкий: источник умеет только «дать файл с
    колонками» и назвать себя. Всё, что с файлом делается дальше —
    проверка схемы, разбиение на батчи, — едино для всех источников
    и живёт не здесь. Иначе добавление второго источника означало бы
    правку всех этапов конвейера.
    """

    name: str
    #: Путь к набору данных. Распаковывается, если это архив.
    path: str
    #: Куда распаковать архив, если он распаковывается.
    extracted_to: str | None = None
    #: Префикс имён батчей. Обязателен, когда источников больше одного:
    #:   два набора за один и тот же месяц переписали бы друг друга.
    prefix: str = "batch_"
    #: Доля источника в потоке, доля единицы — весь поток.
    share: float = 1.0
    enabled: bool = True

    def resolved_path(self, root: Path) -> Path:
        return root / self.path

    def resolved_extracted_to(self, root: Path) -> Path:
        target = self.extracted_to or (self.path.rsplit(".", 1)[0] + ".csv")
        return root / target

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "extracted_to": self.extracted_to,
            "prefix": self.prefix,
            "share": self.share,
            "enabled": self.enabled,
        }


class SourceError(RuntimeError):
    """Источник не найден, не читается или не содержит CSV."""


def sources_from_config(config: Config) -> list[DataSource]:
    """Собрать список источников из конфигурации.

    Если секция `sources` не задана, работает прежняя пара ключей
    `source_zip` / `source_csv`. Обратная совместимость здесь не
    из вежливости: конфигурация уже развёрнута в репозитории, в отчётах
    и в состоянии стоит её хэш, и молча переименовать ключи значило бы
    рассинхронизировать все накопленные артефакты.
    """
    declared = config.get("sources")
    if not declared:
        zip_key = "source_zip" in config.paths
        return [DataSource(
            name="default",
            path=str(config.paths.get("source_zip" if zip_key else "source_csv")),
            extracted_to=str(config.paths["source_csv"]) if zip_key else None,
            prefix=str(config["batching"].get("prefix", "batch_")),
        )]

    if not isinstance(declared, list):
        raise SourceError("sources должна быть списком источников")

    result: list[DataSource] = []
    for position, item in enumerate(declared):
        if not isinstance(item, dict):
            raise SourceError(f"источник №{position}: ожидался словарь")
        name = str(item.get("name", "")).strip()
        path = str(item.get("path", "")).strip()
        if not name:
            raise SourceError(f"источник №{position}: не задано имя")
        if not path:
            raise SourceError(f"источник {name!r}: не задан путь")

        share = float(item.get("share", 1.0))
        if not 0 < share <= 1:
            raise SourceError(
                f"источник {name!r}: доля {share} вне диапазона (0, 1]"
            )

        result.append(DataSource(
            name=name,
            path=path,
            extracted_to=item.get("extracted_to"),
            prefix=str(item.get("prefix", f"{name}_batch_")),
            share=share,
            enabled=bool(item.get("enabled", True)),
        ))

    active = [item for item in result if item.enabled]
    if not active:
        raise SourceError("не осталось ни одного включённого источника")
    if len(active) > 1:
        prefixes = [item.prefix for item in active]
        if len(set(prefixes)) != len(prefixes):
            raise SourceError(
                f"у включённых источников совпадают префиксы имён батчей: "
                f"{prefixes}. Батчи одного месяца из разных наборов "
                f"перезаписали бы друг друга."
            )
    return result


def materialise(source: DataSource, root: Path) -> Path:
    """Получить читаемый CSV источника, распаковав архив при нужде.

    Returns:
        Путь к CSV, готовому к чтению.

    Raises:
        SourceError: файла нет, в архиве нет CSV, распаковать не удалось.
    """
    raw = source.resolved_path(root)
    if not raw.is_file():
        raise SourceError(
            f"источник {source.name!r}: файл не найден — {raw}. "
            f"Проверьте секцию sources в конфигурации."
        )

    if raw.suffix.lower() != ".zip":
        return raw

    target = source.resolved_extracted_to(root)
    if target.is_file():
        return target

    logger.info("Распаковка %s -> %s", raw.name, target.name)
    try:
        with zipfile.ZipFile(raw) as archive:
            members = [
                item for item in archive.namelist()
                if item.lower().endswith(".csv")
            ]
            if not members:
                raise SourceError(
                    f"источник {source.name!r}: в архиве {raw.name} нет CSV"
                )
            if len(members) > 1:
                logger.warning(
                    "в архиве %s %d CSV-файлов, используется первый: %s",
                    raw.name, len(members), members[0],
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(members[0]) as source_stream, open(
                target, "wb"
            ) as destination:
                destination.write(source_stream.read())
    except zipfile.BadZipFile as error:
        raise SourceError(
            f"источник {source.name!r}: {raw.name} не является архивом zip: {error}"
        ) from error
    except OSError as error:
        raise SourceError(
            f"источник {source.name!r}: не удалось распаковать {raw.name}: {error}"
        ) from error

    return target


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
    prepared = prepare_sources(config, force=force)
    if not prepared:
        raise SourceError("не удалось подготовить ни одного источника")
    return prepared[0].path


@dataclass(frozen=True)
class PreparedSource:
    """Источник, готовый к разбиению: CSV на диске и его хэш."""

    source: DataSource
    path: Path
    hash: str | None


def prepare_sources(
    config: Config, force: bool = False
) -> list[PreparedSource]:
    """Подготовить все включённые источники к работе.

    Ошибка одного источника **не отменяет** остальные: набор, из-за
    которого не читается один файл, не должен лишать систему второго.
    Источник, который не удалось подготовить, пропускается с явным
    предупреждением, а состав попавшего в прогон виден по именам
    батчей и по манифесту.
    """
    result: list[PreparedSource] = []
    for source in sources_from_config(config):
        if not source.enabled:
            logger.info("Источник %s выключен", source.name)
            continue
        try:
            path = materialise(source, config.root)
        except SourceError as error:
            logger.warning("%s", error)
            continue
        digest = _file_hash(path) if path.is_file() else None
        logger.info(
            "Источник %s: %s (хэш %s, доля %.2f)",
            source.name, path.name, digest or "н/д", source.share,
        )
        result.append(PreparedSource(source=source, path=path, hash=digest))
    return result


def combined_dataset_hash(config: Config) -> str | None:
    """Хэш всего потока: составляющие вклад поимённо.

    Хэш набора данных должен меняться при изменении **любого** из
    источников. Хэш только первого источника остался бы прежним при
    добавлении второго, а это неверно: выборка изменилась, значит
    и прогресс обучения сохранять нельзя.
    """
    parts: list[str] = []
    for source in sources_from_config(config):
        path = source.resolved_path(config.root)
        if not path.is_file():
            continue
        parts.append(f"{source.name}:{_file_hash(path)}")
    if not parts:
        return None
    digest = hashlib.sha256("|".join(sorted(parts)).encode("utf-8"))
    return digest.hexdigest()[:16]


def dataset_hash(config: Config) -> str | None:
    """Хэш всего потока данных; `None`, если ни один источник не найден.

    Считается по всем источникам, а не по первому: выборка изменилась,
    если изменился любой её источник, иначе при добавлении второго
    набора хэш остался бы прежним и прогресс обучения сохранился бы
    для уже не той выборки.
    """
    combined = combined_dataset_hash(config)
    if combined:
        return combined
    for key in ("source_csv", "source_zip"):
        if key in config.paths:
            path = config.path(key)
            if path.is_file():
                return _file_hash(path)
    return None


# ---------------------------------------------------------------------------
# Разбиение на батчи
# ---------------------------------------------------------------------------


def split_into_batches(
    config: Config,
    source_path: Path,
    store: StateStore,
    reset_progress: bool = True,
    prefix: str | None = None,
) -> list[str]:
    """Разбить один источник на батчи по временной колонке.

    Батчи именуются по периоду: `<prefix>YYYY-MM.csv` и содержат все
    колонки исходного набора. Очистка применяется позже, на этапе
    `update`, чтобы хранилище сырых данных оставалось сырым.

    Args:
        config: конфигурация.
        source_path: распакованный исходный CSV.
        store: состояние прогона, в которое попадёт список батчей.
        reset_progress: сбрасывать ли отметку о пройденных батчах, если
            список батчей не изменился. По умолчанию — да, но
            конвейер передаёт `False`: повторный `init` на тех же данных
            не должен откатывать уже накопленное обучение, иначе
            инкрементальный сценарий CI работает как обучение с нуля.
        prefix: префикс имён батчей. Обязателен при нескольких
            источниках, иначе батчи одного месяца перезапишут друг
            друга. По умолчанию берётся из конфигурации.

    Returns:
        Список относительных путей к батчам в хронологическом порядке.
    """
    time_column = config.time_column
    date_format = config["batching"].get("date_format")
    period = config["batching"].get("size", "month")
    batch_prefix = prefix or str(config["batching"].get("prefix", "batch_"))

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
        path = batches_dir / f"{batch_prefix}{value}.csv"
        payload.to_csv(path, index=False)
        written.append(str(path.relative_to(config.root)))
        logger.info("Батч %s: %d строк", path.name, len(payload))

    store.init_batches(
        written,
        dataset_hash=dataset_hash(config),
        reset=reset_progress,
    )
    logger.info("Создано батчей: %d (период: %s)", len(written), period)
    return written


def _resolve_time_period(period: str) -> bool:
    return period in ("month", "quarter", "year")


def split_all_sources(
    config: Config,
    store: StateStore,
    reset_progress: bool = True,
) -> list[str]:
    """Разбить на батчи все включённые источники и объединить в поток.

    Батчи разных источников попадают в общий список, упорядоченный по
    периоду. Порядок важен не для красоты: `update` идёт по списку
    подряд, и разнородный порядок сбил бы хронологию обучения — сначала
    один набор до конца, потом второй с начала, вместо чередования по
    месяцам.

    Состояние обновляется **один раз на весь поток**. Промежуточная
    запись в состояние затирала бы первый источник вторым, и в
    `state.json` попал бы только один набор — тихо и незаметно.

    Returns:
        Список относительных путей ко всем батчам по возрастанию периода.

    Raises:
        SourceError: ни один источник не удалось подготовить.
    """
    prepared = prepare_sources(config)
    if not prepared:
        raise SourceError(
            "Ни один источник не подготовлен: проверьте секцию sources "
            "и наличие файлов."
        )

    everything: list[tuple[str, str]] = []
    for item in prepared:
        written = _split_without_state(config, item.path, item.source.prefix)
        for relative in written:
            everything.append((relative, Path(relative).stem))

    if len(prepared) == 1:
        ordered = [relative for relative, _ in everything]
        store.init_batches(
            ordered,
            dataset_hash=dataset_hash(config),
            reset=reset_progress,
        )
        return ordered

    ordered = [
        relative for relative, _ in sorted(everything, key=lambda pair: pair[1])
    ]
    store.init_batches(
        ordered,
        dataset_hash=dataset_hash(config),
        reset=reset_progress,
    )
    logger.info(
        "Батчей из %d источников: %d (доли: %s)",
        len(prepared), len(ordered),
        ", ".join(f"{item.source.name}={item.source.share}" for item in prepared),
    )
    return ordered


def _split_without_state(
    config: Config, source_path: Path, prefix: str
) -> list[str]:
    """Разбить источник на файлы, не трогая состояние прогона.

    Отдельная функция, потому что `split_into_batches` пишет в
    состояние: при обработке второго источника это затерело бы запись
    первого.
    """
    time_column = config.time_column
    date_format = config["batching"].get("date_format")
    period = config["batching"].get("size", "month")

    logger.info("Чтение источника %s", source_path.name)
    frame = pd.read_csv(
        source_path, parse_dates=[time_column], date_format=date_format,
        low_memory=False,
    )
    if time_column not in frame.columns:
        raise KeyError(
            f"В источнике {source_path.name} нет колонки {time_column!r}"
        )

    frame = frame.dropna(subset=[time_column]).sort_values(time_column)
    frame["__period__"] = getattr(frame[time_column].dt, f"to_period")(
        "M" if period == "month" else "Q" if period == "quarter" else "Y"
    )

    batches_dir = config.batches_dir
    batches_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for value, group in frame.groupby("__period__", sort=True):
        payload = group.drop(columns=["__period__"])
        path = batches_dir / f"{prefix}{value}.csv"
        payload.to_csv(path, index=False)
        written.append(str(path.relative_to(config.root)))
        logger.info(
            "Батч %s: %d строк (источник с префиксом %r)",
            path.name, len(payload), prefix,
        )
    return written


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
