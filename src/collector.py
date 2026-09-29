"""Сериализуемый сборщик данных (задание 2, пункт 2.b.ii).

Требование: «сохранение модели сборщика данных, допускающего загрузку
актуальных данных извне». Раньше загрузку данных выполняла свободная
функция `data_collection.read_batch`, а путь к батчам брался из YAML и
лежал в `state.json`. Из этого нельзя было сделать артефакт: при загрузке
в другом окружении пришлось бы заново разбирать конфигурацию.

Здесь объект-коллектор, который:

* **сериализуется через `pickle`** целиком, вместе с каталогом, шаблоном
  имени и требуемой схемой колонок — открытые данные, без ссылок на
  внутренние объекты конвейера;
* **обнаруживает батчи сам**, а не полагается на запись в состоянии;
* **принимает новый файл извне** (`append_external`): проверяет схему,
  разбирает даты, отказывается принимать уже известный период и сообщает,
  что именно не так, если файл не подходит.

Объект можно положить в артефакт CI и использовать в другом прогоне:

    collector = pickle.load(open("collector.pkl", "rb"))
    collector.append_external("data/new/batch_2018-07.csv")

Модуль запускается и как скрипт — это позволяет проверить сборщик
в CI отдельным шагом:

    python -m src.collector --describe
    python -m src.collector --append data/new/batch_2018-07.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.utils import parse_datetime, to_json_serializable

logger = logging.getLogger(__name__)

#: Шаблон имени батча: префикс, затем период, затем расширение.
#: Период обязан выглядеть как период: `YYYY-MM` (месяц), `YYYY-Qn`
#: (квартал) или `YYYY` (год). Раньше подходил любой букво-цифровой
#: хвост, и файл вроде `batch_summary.csv` принимался за батч.
BATCH_PATTERN = re.compile(
    r"^batch_(?P<period>\d{4}(?:-(?:\d{2}|Q[1-4]))?)\.csv$"
)

#: Расширения, из которых новый батч принимается.
ACCEPTED_SUFFIXES = (".csv",)


class CollectorError(RuntimeError):
    """Файл не подходит для приёма в качестве батча."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class BatchCollector:
    """Обнаружение батчей и приём новых данных извне.

    Атрибуты подобраны так, чтобы объект оставался пригодным для
    `pickle`: только пути, строки, числа и списки примитивов.
    """

    batches_dir: str
    time_column: str = "INSR_BEGIN"
    date_format: str | None = None
    required_columns: list[str] = field(default_factory=list)
    prefix: str = "batch_"

    # ------------------------------------------------------------------
    # Обнаружение
    # ------------------------------------------------------------------

    def directory(self) -> Path:
        return Path(self.batches_dir)

    def batch_names(self) -> list[str]:
        """Имена батчей, отсортированные по периоду."""
        if not self.directory().is_dir():
            return []
        found: list[tuple[str, str]] = []
        for item in self.directory().iterdir():
            if not item.is_file():
                continue
            match = BATCH_PATTERN.match(item.name)
            if match and item.name.startswith(self.prefix):
                found.append((match.group("period"), item.name))
        return [name for _, name in sorted(found)]

    def batch_paths(self) -> list[Path]:
        return [self.directory() / name for name in self.batch_names()]

    def periods(self) -> list[str]:
        """Периоды батчей в хронологическом порядке."""
        result: list[str] = []
        for name in self.batch_names():
            match = BATCH_PATTERN.match(name)
            if match:
                result.append(match.group("period"))
        return result

    def __len__(self) -> int:
        return len(self.batch_names())

    def has_batch(self, period: str) -> bool:
        return period in self.periods()

    def path_for(self, period: str) -> Path:
        return self.directory() / f"{self.prefix}{period}.csv"

    # ------------------------------------------------------------------
    # Приём данных извне
    # ------------------------------------------------------------------

    def validate(self, frame: pd.DataFrame, source: str) -> dict[str, Any]:
        """Проверить файл-кандидат на пригодность.

        Returns:
            Сведения о проверке; при успехе `ok` равен True.

        Raises:
            CollectorError: файл не подходит, с указанием причины.
        """
        if frame.empty:
            raise CollectorError(f"{source}: файл не содержит строк")

        missing = [
            column for column in self.required_columns if column not in frame.columns
        ]
        if missing:
            raise CollectorError(
                f"{source}: отсутствуют обязательные колонки: {missing}"
            )

        if self.time_column not in frame.columns:
            raise CollectorError(
                f"{source}: нет временной колонки {self.time_column!r}, "
                f"по ней определяется период батча"
            )

        parsed = parse_datetime(frame[self.time_column], self.date_format)
        unparsed = int(parsed.isna().sum())
        if unparsed == len(parsed):
            raise CollectorError(
                f"{source}: не удалось разобрать ни одной даты в "
                f"{self.time_column!r}"
            )
        if unparsed:
            logger.warning(
                "%s: %d из %d дат не разобраны, они будут исключены",
                source, unparsed, len(parsed),
            )

        periods = sorted(
            {
                value
                for value in parsed.dropna().dt.strftime("%Y-%m").unique()
            }
        )
        if len(periods) > 1:
            raise CollectorError(
                f"{source}: файл содержит {len(periods)} разных периодов "
                f"({periods}). Один батч — один период; разделите файл."
            )

        return {
            "ok": True,
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "period": periods[0],
            "unparsed_dates": unparsed,
        }

    def append_external(
        self, path: str | Path, overwrite: bool = False
    ) -> dict[str, Any]:
        """Принять новый батч из внешнего файла.

        Args:
            path: файл с данными. Формат — те же колонки, что у батчей.
            overwrite: разрешить замену уже существующего батча.

        Returns:
            Сведения о принятом батче, включая итоговый путь.

        Raises:
            CollectorError: файл не существует, не читается, не проходит
                проверку схемы либо такой период уже принят.
        """
        source = Path(path)
        if not source.is_file():
            raise CollectorError(f"Файл не найден: {source}")
        if source.suffix.lower() not in ACCEPTED_SUFFIXES:
            raise CollectorError(
                f"{source.name}: поддерживаются только "
                f"{', '.join(ACCEPTED_SUFFIXES)}, получено {source.suffix}"
            )

        frame = pd.read_csv(source, low_memory=False)
        verdict = self.validate(frame, source.name)

        period = verdict["period"]
        target = self.path_for(period)
        if target.exists() and not overwrite:
            raise CollectorError(
                f"Батч за период {period} уже принят ({target.name}). "
                f"Чтобы заменить его, передайте overwrite=True."
            )

        # Время приводится к датам и записывается в едином виде, иначе
        # следующий прогон разбирал бы его заново и не смог бы отличить
        # формат источника от собственного.
        payload = frame.copy()
        if verdict["unparsed_dates"]:
            payload = payload.loc[
                parse_datetime(payload[self.time_column], self.date_format).notna()
            ]
        payload[self.time_column] = payload[self.time_column].astype("datetime64[ns]")
        payload.to_csv(target, index=False)

        result = {
            **verdict,
            "source": str(source),
            "target": str(target),
            "rows_written": int(len(payload)),
            "total_batches": len(self),
            "registered_at": _utc_now(),
        }
        logger.info(
            "Принят батч %s: %d строк (всего батчей: %d)",
            period, result["rows_written"], result["total_batches"],
        )
        return result

    # ------------------------------------------------------------------
    # Описание и сериализация
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Сводка по коллектору — идёт в артефакт и в отчёт.

        Подсчёт строк читает все батчи, поэтому на полном наборе это
        около 50 МБ ввода-вывода. Берётся только первая колонка: для
        числа строк остальные не нужны, а читать 16 колонок в 8–10 раз
        дороже.
        """
        names = self.batch_names()
        periods = self.periods()
        rows: list[int] = []
        for path in self.batch_paths():
            try:
                rows.append(sum(
                    len(chunk)
                    for chunk in pd.read_csv(
                        path, usecols=[0], chunksize=50_000
                    )
                ))
            except (OSError, ValueError, IndexError) as error:
                logger.warning("Не удалось посчитать строки в %s: %s", path.name, error)
                rows.append(0)
        return {
            "batches_dir": str(self.directory()),
            "exists": self.directory().is_dir(),
            "time_column": self.time_column,
            "date_format": self.date_format,
            "required_columns": list(self.required_columns),
            "n_batches": len(names),
            "first_batch": names[0] if names else None,
            "last_batch": names[-1] if names else None,
            "first_period": periods[0] if periods else None,
            "last_period": periods[-1] if periods else None,
            "total_rows": sum(rows),
            "serializable": True,
        }

    def save(self, path: str | Path) -> Path:
        """Атомарно сохранить коллектор для выгрузки как артефакта."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            with open(temporary, "wb") as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        logger.info("Сборщик сохранён: %s", target.name)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "BatchCollector":
        """Загрузить коллектор из артефакта."""
        source = Path(path)
        if not source.is_file():
            raise CollectorError(f"Файл сборщика не найден: {source}")
        with open(source, "rb") as handle:
            collector = pickle.load(handle)
        if not isinstance(collector, cls):
            raise CollectorError(
                f"{source.name}: ожидался BatchCollector, "
                f"получен {type(collector).__name__}"
            )
        return collector

    @classmethod
    def from_config(cls, config: Any) -> "BatchCollector":
        """Собрать коллектор по конфигурации конвейера."""
        return cls(
            # `Config.path` разрешает путь по КЛЮЧУ секции `paths`, а не
            # по значению: значение равно `data/raw_batches`, и передача
            # его как ключа приводит к `ConfigError`.
            batches_dir=str(config.path("data_raw")),
            time_column=config.time_column,
            date_format=config["batching"].get("date_format"),
            required_columns=list(config.get("schema", {}).get("required_columns", [])),
        )


# ---------------------------------------------------------------------------
# Командный интерфейс
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.collector",
        description="Сериализуемый сборщик данных: описание и приём батчей извне",
    )
    parser.add_argument(
        "--describe", action="store_true",
        help="Показать сводку по найденным батчам",
    )
    parser.add_argument(
        "--append", metavar="FILE",
        help="Принять новый батч из внешнего файла",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Разрешить замену существующего батча",
    )
    parser.add_argument(
        "--save", metavar="FILE",
        help="Сохранить сборщик в файл для выгрузки как артефакта",
    )
    parser.add_argument("--json", action="store_true", help="Вывод в формате JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа. Возвращает код завершения."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        stream=sys.stdout,
    )
    args = build_parser().parse_args(argv)

    if not any([args.describe, args.append, args.save]):
        build_parser().print_help()
        return 2

    try:
        from src.config import load_config

        collector = BatchCollector.from_config(load_config())
    except Exception as error:  # noqa: BLE001
        print(f"Не удалось собрать коллектор: {error}", file=sys.stderr)
        return 1

    payload: dict[str, Any] = {}
    try:
        if args.describe:
            payload["describe"] = collector.describe()
        if args.append:
            payload["appended"] = collector.append_external(
                args.append, overwrite=args.overwrite
            )
        if args.save:
            collector.save(args.save)
            payload["saved"] = args.save
    except CollectorError as error:
        if args.json:
            print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False, indent=2))
        else:
            print(f"Ошибка: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(
            to_json_serializable({"ok": True, **payload}), ensure_ascii=False, indent=2
        ))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
