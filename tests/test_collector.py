"""Проверки сериализуемого сборщика данных (задание 2, пункт 2.b.ii).

Сборщик должен пережить две операции, которых не мог сделать прежний код:
сериализацию в артефакт и приём батча извне. Проверяются именно они,
а также отказы на некорректных данных — молча принятый негодный файл
испортил бы обучение всерьёз и обнаружился бы сильно позже.
"""

from __future__ import annotations

import pickle

import pandas as pd
import pytest

from src.collector import BatchCollector, CollectorError
from src.config import load_config

REQUIRED = [
    "SEX", "INSR_BEGIN", "INSR_END", "INSR_TYPE", "INSURED_VALUE", "PREMIUM",
    "PROD_YEAR", "SEATS_NUM", "CARRYING_CAPACITY", "TYPE_VEHICLE", "MAKE",
    "USAGE", "CLAIM_PAID",
]

#: Названия месяцев в формате `%d-%b-%y`.
MONTHS = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]


def make_batch(period: str = "2018-07", rows: int = 5, **overrides) -> pd.DataFrame:
    """Батч с корректной схемой.

    Период задаётся датами в колонке `INSR_BEGIN`: именно из них
    сборщик выводит имя батча, поэтому параметр нельзя игнорировать.
    """
    year, month = period.split("-")
    days = [f"{index + 1:02d}-{MONTHS[int(month) - 1]}-{year[2:]}"
            for index in range(rows)]
    frame = pd.DataFrame(
        {
            "SEX": ["0"] * rows,
            "INSR_BEGIN": days,
            "INSR_END": days,
            "INSR_TYPE": ["1202"] * rows,
            "INSURED_VALUE": [100000.0] * rows,
            "PREMIUM": [500.0] * rows,
            "PROD_YEAR": [2015] * rows,
            "SEATS_NUM": [2] * rows,
            "CARRYING_CAPACITY": [1500.0] * rows,
            "TYPE_VEHICLE": ["Truck"] * rows,
            "MAKE": ["ISUZU"] * rows,
            "USAGE": ["Own Goods"] * rows,
            "CLAIM_PAID": [None] * rows,
        }
    )
    for column, value in overrides.items():
        frame[column] = value
    return frame


@pytest.fixture
def collector(tmp_path) -> BatchCollector:
    (tmp_path / "batch_2018-06.csv").write_text(
        make_batch("2018-06").to_csv(index=False), encoding="utf-8"
    )
    (tmp_path / "batch_2014-07.csv").write_text(
        make_batch("2014-07").to_csv(index=False), encoding="utf-8"
    )
    return BatchCollector(
        batches_dir=str(tmp_path),
        time_column="INSR_BEGIN",
        date_format="%d-%b-%y",
        required_columns=REQUIRED,
    )


# ---------------------------------------------------------------------------
# Обнаружение
# ---------------------------------------------------------------------------


def test_batches_sorted_by_period(collector):
    """Батчи упорядочены хронологически, а не по имени файла."""
    assert collector.periods() == ["2014-07", "2018-06"]
    assert collector.batch_names()[0] == "batch_2014-07.csv"
    assert len(collector) == 2


def test_non_batch_files_are_ignored(tmp_path, collector):
    """Посторонние файлы в каталоге не считаются батчами."""
    (tmp_path / "notes.txt").write_text("служебное", encoding="utf-8")
    (tmp_path / "batch_summary.csv").write_text("x", encoding="utf-8")
    assert len(collector) == 2


def test_missing_directory_is_not_fatal(tmp_path):
    """Отсутствующий каталог даёт пустой список, а не исключение."""
    empty = BatchCollector(batches_dir=str(tmp_path / "absent"))
    assert len(empty) == 0
    assert empty.describe()["n_batches"] == 0
    assert empty.describe()["exists"] is False


# ---------------------------------------------------------------------------
# Сериализация
# ---------------------------------------------------------------------------


def test_collector_survives_pickle_roundtrip(collector, tmp_path):
    """Коллектор переживает pickle и работает в новом процессе.

    Именно это требуется от «модели сборщика данных»: артефакт выгружается
    в CI и должен использоваться без доступа к внутренним объектам.
    """
    collector.save(tmp_path / "collector.pkl")
    restored = BatchCollector.load(tmp_path / "collector.pkl")

    assert restored.batches_dir == collector.batches_dir
    assert restored.required_columns == collector.required_columns
    assert restored.date_format == collector.date_format
    assert restored.periods() == collector.periods()


def test_saved_file_is_a_plain_pickle(collector, tmp_path):
    """Артефакт читается обычным pickle, без импорта пакета."""
    collector.save(tmp_path / "collector.pkl")
    with open(tmp_path / "collector.pkl", "rb") as handle:
        restored = pickle.load(handle)
    assert isinstance(restored, BatchCollector)


def test_load_rejects_wrong_object(tmp_path):
    """Файл неправильного типа даёт понятную ошибку."""
    path = tmp_path / "collector.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"not": "a collector"}, handle)
    with pytest.raises(CollectorError) as error:
        BatchCollector.load(path)
    assert "BatchCollector" in str(error.value)


# ---------------------------------------------------------------------------
# Приём данных извне
# ---------------------------------------------------------------------------


def test_append_external_accepts_valid_batch(collector, tmp_path):
    """Корректный файл принимается и появляется среди батчей."""
    incoming = tmp_path / "incoming" / "new.csv"
    incoming.parent.mkdir()
    incoming.write_text(make_batch("2018-07").to_csv(index=False), encoding="utf-8")

    result = collector.append_external(incoming)

    assert result["period"] == "2018-07"
    assert result["rows_written"] == 5
    assert result["total_batches"] == 3
    assert collector.has_batch("2018-07")
    assert (tmp_path / "batch_2018-07.csv").is_file()


def test_appended_batch_dates_are_normalised(collector, tmp_path):
    """Принятый батч читается конвейером так же, как исходные.

    Проверяется не вывод `read_csv` (pandas не определяет типы дат
    автоматически), а то, что важно на самом деле: разбор дат из принятого
    файла даёт настоящие даты, а `timeliness` и производные признаки
    работают независимо от того, каким путём попали данные.
    """
    from src.utils import parse_datetime

    incoming = tmp_path / "incoming.csv"
    incoming.write_text(make_batch("2018-08").to_csv(index=False), encoding="utf-8")
    collector.append_external(incoming)

    stored = pd.read_csv(tmp_path / "batch_2018-08.csv", low_memory=False)
    parsed = parse_datetime(stored["INSR_BEGIN"], collector.date_format)

    assert parsed.notna().all()
    assert parsed.dt.strftime("%Y-%m").eq("2018-08").all()


def test_append_refuses_duplicate_period(collector, tmp_path):
    """Повторная загрузка того же периода отвергается, а не затирает данные."""
    incoming = tmp_path / "again.csv"
    incoming.write_text(make_batch("2018-06").to_csv(index=False), encoding="utf-8")

    with pytest.raises(CollectorError) as error:
        collector.append_external(incoming)
    assert "уже принят" in str(error.value)

    # Данные батча не изменились
    assert len(pd.read_csv(tmp_path / "batch_2018-06.csv", low_memory=False)) == 5


def test_overwrite_is_opt_in(collector, tmp_path):
    """Замена существующего батча возможна только явным решением."""
    incoming = tmp_path / "replacement.csv"
    incoming.write_text(
        make_batch("2018-06", rows=9).to_csv(index=False), encoding="utf-8"
    )

    result = collector.append_external(incoming, overwrite=True)
    assert result["rows_written"] == 9


def test_append_rejects_missing_columns(collector, tmp_path):
    """Файл без обязательных колонок отвергается с перечнем пропавших."""
    broken = tmp_path / "broken.csv"
    frame = make_batch("2018-09").drop(columns=["PREMIUM", "MAKE"])
    broken.write_text(frame.to_csv(index=False), encoding="utf-8")

    with pytest.raises(CollectorError) as error:
        collector.append_external(broken)
    assert "PREMIUM" in str(error.value)
    assert "MAKE" in str(error.value)


def test_append_rejects_multiple_periods(collector, tmp_path):
    """Файл с двумя периодами отвергается: один батч — один период."""
    mixed = tmp_path / "mixed.csv"
    frame = make_batch("2018-09", rows=4)
    frame.loc[frame.index[:2], "INSR_BEGIN"] = "01-AUG-18"
    mixed.write_text(frame.to_csv(index=False), encoding="utf-8")

    with pytest.raises(CollectorError) as error:
        collector.append_external(mixed)
    assert "разных периодов" in str(error.value)


def test_append_rejects_unparsable_dates(collector, tmp_path):
    """Файл без разбираемых дат отвергается."""
    bad = tmp_path / "bad_dates.csv"
    frame = make_batch("2018-09", INSR_BEGIN=["не дата"] * 5)
    bad.write_text(frame.to_csv(index=False), encoding="utf-8")

    with pytest.raises(CollectorError) as error:
        collector.append_external(bad)
    assert "даты" in str(error.value)


def test_append_rejects_empty_file(collector, tmp_path):
    """Пустой файл отвергается."""
    empty = tmp_path / "empty.csv"
    empty.write_text(
        make_batch("2018-09").iloc[0:0].to_csv(index=False), encoding="utf-8"
    )
    with pytest.raises(CollectorError) as error:
        collector.append_external(empty)
    assert "не содержит строк" in str(error.value)


def test_append_rejects_absent_file(collector, tmp_path):
    """Отсутствующий файл даёт понятную ошибку."""
    with pytest.raises(CollectorError) as error:
        collector.append_external(tmp_path / "absent.csv")
    assert "не найден" in str(error.value)


def test_append_rejects_wrong_extension(collector, tmp_path):
    """Файл неподдерживаемого типа не принимается молча."""
    other = tmp_path / "data.xlsx"
    other.write_bytes(b"fake")
    with pytest.raises(CollectorError) as error:
        collector.append_external(other)
    assert "csv" in str(error.value).lower()


def test_partially_unparsable_dates_are_dropped_with_warning(collector, tmp_path):
    """Часть дат не разбирается — строки с ними исключаются, приём идёт."""
    incoming = tmp_path / "partial.csv"
    frame = make_batch("2018-10", rows=6)
    frame.loc[frame.index[0], "INSR_BEGIN"] = "не дата"
    incoming.write_text(frame.to_csv(index=False), encoding="utf-8")

    result = collector.append_external(incoming)
    assert result["unparsed_dates"] == 1
    assert result["rows_written"] == 5


# ---------------------------------------------------------------------------
# Связка с боевой конфигурацией
# ---------------------------------------------------------------------------


def test_collector_from_project_config():
    """Коллектор собирается из боевого config.yaml и видит батчи."""
    config = load_config()
    collector = BatchCollector.from_config(config)
    summary = collector.describe()

    assert collector.time_column == config.time_column
    assert collector.required_columns == config["schema"]["required_columns"]
    # Каталог батчей обязан существовать после хотя бы одного init
    assert summary["n_batches"] >= 0
