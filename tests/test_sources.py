"""Проверки нескольких источников данных (1.b.ii).

Требование: «конфигурируемый список источников, единый контракт загрузки».
Проверяется не сам факт списка, а то, что он не приносит скрытых
потерь: сбой одного источника не должен лишать систему остальных,
батчи разных наборов не должны перезаписывать друг друга, а хэш данных
меняться при изменении **любого** источника.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.config import Config
from src.data_collection import (
    DataSource,
    SourceError,
    combined_dataset_hash,
    materialise,
    prepare_sources,
    sources_from_config,
    split_all_sources,
)
from src.state import StateStore

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

COLUMNS = [
    "SEX", "INSR_BEGIN", "INSR_TYPE", "INSURED_VALUE", "PREMIUM",
    "CLAIM_PAID",
]


def make_csv(path: Path, months: list[int], marker: str) -> None:
    """CSV с заданными месяцами; `marker` попадает в MAKE-подобную колонку."""
    rows = []
    for month in months:
        for day in range(1, 4):
            rows.append({
                "SEX": "0",
                "INSR_BEGIN": f"{day:02d}-{MONTHS[month - 1]}-18",
                "INSR_TYPE": marker,
                "INSURED_VALUE": 100000.0,
                "PREMIUM": 500.0,
                "CLAIM_PAID": None,
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[COLUMNS].to_csv(path, index=False)


def make_config(tmp_path: Path, sources: list[dict]) -> Config:
    return Config(
        data={
            "paths": {
                "data_raw": "raw_batches",
                "models": "models",
                "reports": "reports",
                "metadata": "metadata",
                "state_file": "state.json",
                "rules_file": "rules.json",
            },
            "batching": {
                "size": "month",
                "time_column": "INSR_BEGIN",
                "date_format": "%d-%b-%y",
                "prefix": "batch_",
            },
            "target": {"column": "CLAIM_PAID", "name": "HAS_CLAIM"},
            "sources": sources,
        },
        root=tmp_path,
    )


@pytest.fixture
def two_sources(tmp_path):
    """Два набора: первый — январь-март, второй — апрель-июнь."""
    make_csv(tmp_path / "data" / "a.csv", [1, 2, 3], "A")
    make_csv(tmp_path / "data" / "b.csv", [4, 5, 6], "B")
    return make_config(tmp_path, [
        {"name": "a", "path": "data/a.csv", "prefix": "a_batch_", "share": 0.6},
        {"name": "b", "path": "data/b.csv", "prefix": "b_batch_", "share": 0.4},
    ])


# ---------------------------------------------------------------------------
# Разбор конфигурации
# ---------------------------------------------------------------------------


def test_sources_are_read_from_config(two_sources):
    """Список источников и доли берутся из конфигурации."""
    sources = sources_from_config(two_sources)

    assert [item.name for item in sources] == ["a", "b"]
    assert [item.share for item in sources] == [0.6, 0.4]
    assert [item.prefix for item in sources] == ["a_batch_", "b_batch_"]


def test_missing_section_falls_back_to_legacy_keys(tmp_path):
    """Без секции `sources` работает прежняя пара ключей.

    Обратная совместимость здесь не из вежливости: в репозитории уже
    лежит конфигурация, а её хэш стоит в отчётах и состоянии.
    """
    config = Config(
        data={
            "paths": {
                "source_zip": "data/s.zip",
                "source_csv": "data/s.csv",
                "data_raw": "raw", "models": "m", "reports": "r",
                "metadata": "md", "state_file": "state.json",
                "rules_file": "rules.json",
            },
            "batching": {"prefix": "batch_"},
        },
        root=tmp_path,
    )
    sources = sources_from_config(config)

    assert len(sources) == 1
    assert sources[0].path == "data/s.zip"


def test_identical_prefixes_are_rejected(tmp_path):
    """Совпадающие префиксы запрещены: батчи перезаписали бы друг друга."""
    make_csv(tmp_path / "data" / "a.csv", [1], "A")
    make_csv(tmp_path / "data" / "b.csv", [1], "B")
    config = make_config(tmp_path, [
        {"name": "a", "path": "data/a.csv", "prefix": "batch_"},
        {"name": "b", "path": "data/b.csv", "prefix": "batch_"},
    ])

    with pytest.raises(SourceError) as error:
        sources_from_config(config)
    assert "префиксы" in str(error.value)


def test_share_out_of_range_is_rejected(tmp_path):
    """Доля вне (0, 1] — ошибка конфигурации, а не молчаливое исправление."""
    make_csv(tmp_path / "data" / "a.csv", [1], "A")
    config = make_config(tmp_path, [
        {"name": "a", "path": "data/a.csv", "prefix": "a_", "share": 0},
    ])

    with pytest.raises(SourceError) as error:
        sources_from_config(config)
    assert "доля" in str(error.value)


def test_all_sources_disabled_is_an_error(tmp_path):
    """Если включённых источников нет, это ошибка, а не пустой поток."""
    config = make_config(tmp_path, [
        {"name": "a", "path": "data/a.csv", "enabled": False},
    ])

    with pytest.raises(SourceError) as error:
        sources_from_config(config)
    assert "включённого источника" in str(error.value)


def test_source_without_name_or_path_is_rejected(tmp_path):
    """Источник без имени или пути не проходит: ошибка видна сразу."""
    for bad in ({"path": "data/a.csv"}, {"name": "a"}):
        config = make_config(tmp_path, [bad])
        with pytest.raises(SourceError):
            sources_from_config(config)


# ---------------------------------------------------------------------------
# Подготовка
# ---------------------------------------------------------------------------


def test_zip_source_is_unpacked(tmp_path):
    """Архив распаковывается, и дальше используется CSV."""
    raw = tmp_path / "data" / "a.csv"
    make_csv(raw, [1], "A")
    archive = tmp_path / "data" / "a.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(raw, arcname="a.csv")
    raw.unlink()

    source = DataSource(
        name="a", path="data/a.zip", extracted_to="data/out.csv"
    )
    result = materialise(source, tmp_path)

    assert result.is_file()
    assert result.name == "out.csv"


def test_missing_source_file_gives_a_clear_error(tmp_path):
    """Отсутствующий файл назван поимённо, а не «ошибка загрузки»."""
    source = DataSource(name="missing", path="data/nope.csv")
    with pytest.raises(SourceError) as error:
        materialise(source, tmp_path)
    assert "missing" in str(error.value)
    assert "nope.csv" in str(error.value)


def test_broken_archive_is_reported_as_such(tmp_path):
    """Файл с расширением zip, но не являющийся архивом, распознаётся."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "fake.zip").write_bytes(b"not an archive")
    source = DataSource(name="fake", path="data/fake.zip")

    with pytest.raises(SourceError) as error:
        materialise(source, tmp_path)
    assert "zip" in str(error.value)


def test_one_broken_source_does_not_block_the_others(two_sources, tmp_path):
    """Сбой одного источника не лишает систему второго.

    Иначе отсутствующий дополнительный файл останавливал бы обучение
    на наборе, который полностью пригоден.
    """
    (tmp_path / "data" / "a.csv").unlink()
    prepared = prepare_sources(two_sources)

    assert [item.source.name for item in prepared] == ["b"]


def test_disabled_source_is_skipped(two_sources, tmp_path):
    """Выключенный источник не читается, даже если файл на месте."""
    config = make_config(two_sources.root, [
        {"name": "a", "path": "data/a.csv", "prefix": "a_batch_"},
        {"name": "b", "path": "data/b.csv", "prefix": "b_batch_",
         "enabled": False},
    ])
    prepared = prepare_sources(config)
    assert [item.source.name for item in prepared] == ["a"]


# ---------------------------------------------------------------------------
# Хэш потока
# ---------------------------------------------------------------------------


def test_hash_changes_when_any_source_changes(two_sources, tmp_path):
    """Изменение второго источника меняет хэш всего потока.

    Хэш только первого источника остался бы прежним, и прогресс
    обучения сохранился бы для уже не той выборки.
    """
    before = combined_dataset_hash(two_sources)
    make_csv(tmp_path / "data" / "b.csv", [4, 5, 7], "B")
    after = combined_dataset_hash(two_sources)

    assert before != after


def test_hash_is_stable_when_nothing_changes(two_sources):
    """Хэш не «прыгает» сам по себе: иначе прогресс терялся бы зря."""
    assert combined_dataset_hash(two_sources) == combined_dataset_hash(two_sources)


# ---------------------------------------------------------------------------
# Объединение потока
# ---------------------------------------------------------------------------


def test_both_sources_land_in_one_ordered_stream(two_sources, tmp_path):
    """Батчи обоих наборов попадают в один список по возрастанию периода."""
    store = StateStore(tmp_path / "state.json", root=tmp_path)
    written = split_all_sources(two_sources, store)

    names = [Path(item).stem for item in written]
    assert len(names) == 6
    # Периоды возрастают: месяцы первого набора раньше второго
    months = [int(name.rsplit("-", 1)[-1]) for name in names]
    assert months == sorted(months)
    assert any(name.startswith("a_batch_") for name in names)
    assert any(name.startswith("b_batch_") for name in names)


def test_all_batches_reach_the_state(two_sources, tmp_path):
    """В состояние попадают батчи обоих наборов, а не только последнего.

    Промежуточная запись в состояние затирала бы первый источник
    вторым, и это обнаружилось бы как «пропали батчи».
    """
    store = StateStore(tmp_path / "state.json", root=tmp_path)
    written = split_all_sources(two_sources, store)

    assert store.batches == written
    assert len(store.batches) == 6


def test_batches_are_not_overwritten(two_sources, tmp_path):
    """Файлы разных наборов сосуществуют: префиксы не дают им слиться."""
    store = StateStore(tmp_path / "state.json", root=tmp_path)
    split_all_sources(two_sources, store)

    files = sorted(p.name for p in (tmp_path / "raw_batches").glob("*.csv"))
    assert len(files) == 6
    assert len({name[:8] for name in files}) == 2


def test_single_source_keeps_prefix_from_config(two_sources, tmp_path):
    """С одним источником имена батчей остаются прежними."""
    config = make_config(two_sources.root, [
        {"name": "a", "path": "data/a.csv", "prefix": "batch_"},
    ])
    store = StateStore(tmp_path / "state2.json", root=tmp_path)
    written = split_all_sources(config, store)

    assert all(Path(item).name.startswith("batch_") for item in written)
    assert len(written) == 3


def test_no_usable_source_is_reported(two_sources, tmp_path):
    """Если не подготовлен ни один источник, это явная ошибка."""
    (tmp_path / "data" / "a.csv").unlink()
    (tmp_path / "data" / "b.csv").unlink()
    store = StateStore(tmp_path / "state3.json", root=tmp_path)

    with pytest.raises(SourceError) as error:
        split_all_sources(two_sources, store)
    assert "sources" in str(error.value)
