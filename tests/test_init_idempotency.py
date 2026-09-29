"""Проверки того, что `init` не откатывает накопленное обучение.

Сценарий, ради которого это проверяется: CI вызывает `init` на каждом
запуске, в том числе по расписанию. Набор батчей при этом не меняется,
а каталог с самими батчами каждый раз создаётся заново — `data/raw_batches/`
не входит в кэш состояния. Если бы `init` сбрасывал прогресс, каждое
срабатывание расписания обучало бы систему с первого батча, и
дообучение (пункт 3.5 задания 2, 3 балла) было бы невозможно.

Прогресс должен сбрасываться ровно в двух случаях: набор данных
изменился или сброс запрошен явно.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.config import Config
from src.data_collection import split_into_batches
from src.state import StateStore


# ---------------------------------------------------------------------------
# Подготовка
# ---------------------------------------------------------------------------


MONTHS = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]

COLUMNS = [
    "SEX", "INSR_BEGIN", "INSR_END", "INSR_TYPE", "INSURED_VALUE", "PREMIUM",
    "PROD_YEAR", "SEATS_NUM", "CARRYING_CAPACITY", "TYPE_VEHICLE", "MAKE",
    "USAGE", "CLAIM_PAID",
]


def make_source(path: Path, periods: int = 3) -> None:
    """Исходный CSV на `periods` месяцев, в формате источника `%d-%b-%y`."""
    rows = []
    for month in range(1, periods + 1):
        for day in range(1, 6):
            rows.append({
                "SEX": "0",
                "INSR_BEGIN": f"{day:02d}-{MONTHS[month - 1]}-18",
                "INSR_END": f"{day:02d}-{MONTHS[month - 1]}-19",
                "INSR_TYPE": "1202",
                "INSURED_VALUE": 100000.0 + day,
                "PREMIUM": 500.0,
                "PROD_YEAR": 2015,
                "SEATS_NUM": 2,
                "CARRYING_CAPACITY": 1500.0,
                "TYPE_VEHICLE": "Truck",
                "MAKE": "ISUZU",
                "USAGE": "Own Goods",
                "CLAIM_PAID": None,
            })
    pd.DataFrame(rows)[COLUMNS].to_csv(path, index=False)


def make_project(tmp_path: Path) -> tuple[Config, Path]:
    """Готовый проект: архив с данными, распакованный CSV, конфигурация."""
    csv_path = tmp_path / "data" / "source.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    make_source(csv_path)

    zip_path = tmp_path / "data" / "source.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(csv_path, arcname="source.csv")

    data = {
        "paths": {
            "source_zip": "data/source.zip",
            "source_csv": "data/source.csv",
            "data_raw": "raw_batches",
            "data_processed": "processed",
            "models": "models",
            "reports": "reports",
            "metadata": "metadata",
            "rules_file": "rules.json",
            "state_file": "state.json",
            "site": "site",
        },
        "batching": {"size": "month", "time_column": "INSR_BEGIN",
                     "date_format": "%d-%b-%y"},
        "schema": {"required_columns": COLUMNS},
        "run": {"seed": 42},
    }
    config = Config(data=data, root=tmp_path)
    config.ensure_dirs()
    return config, csv_path


# ---------------------------------------------------------------------------
# Идемпотентность повторного разбиения
# ---------------------------------------------------------------------------


def test_repeat_split_keeps_progress(tmp_path):
    """Повторное разбиение тех же данных не сбрасывает прогресс.

    Без этого `init` в CI откатывал бы обучение на каждом запуске.
    """
    from src.data_collection import extract_source, dataset_hash

    config, _ = make_project(tmp_path)
    store = StateStore(config.state_file, root=config.root)

    source = extract_source(config)
    written = split_into_batches(config, source, store, reset_progress=False)
    store.mark_processed(0)
    store.mark_processed(1)

    # Каталог с батчами удалён — как в CI, где он не попадает в кэш
    for path in (config.root / name for name in written):
        Path(path).unlink()

    source = extract_source(config)
    again = split_into_batches(config, source, store, reset_progress=False)

    assert again == written
    assert store.last_processed == 1
    assert len(store.history) == 2


def test_explicit_reset_starts_from_scratch(tmp_path):
    """Явный сброс откатывает прогресс, даже если данные те же."""
    from src.data_collection import extract_source

    config, _ = make_project(tmp_path)
    store = StateStore(config.state_file, root=config.root)

    source = extract_source(config)
    written = split_into_batches(config, source, store, reset_progress=False)
    store.mark_processed(0)

    source = extract_source(config)
    was_reset = store.init_batches(
        written, dataset_hash=None, reset=True
    )

    assert was_reset is True
    assert store.last_processed == -1
    assert store.history == []


def test_changed_dataset_resets_progress(tmp_path):
    """Другие данные в том же наборе батчей откатывают прогресс.

    Номера батчей остаются прежними, но строки в них другие, поэтому
    обученная на старых данных модель не соответствует текущим данным.
    """
    from src.data_collection import extract_source, dataset_hash

    config, csv_path = make_project(tmp_path)
    store = StateStore(config.state_file, root=config.root)

    source = extract_source(config)
    written = split_into_batches(config, source, store, reset_progress=False)
    store.mark_processed(0)

    # Данные изменились, месяцы те же
    make_source(csv_path, periods=3)
    source = extract_source(config)
    new_written = split_into_batches(
        config, source, store, reset_progress=True
    )

    assert new_written == written, "список батчей совпал, но это неважно"
    assert store.last_processed == -1


def test_history_is_kept_when_progress_kept(tmp_path):
    """История прогонов не теряется при сохранении прогресса."""
    from src.data_collection import extract_source

    config, _ = make_project(tmp_path)
    store = StateStore(config.state_file, root=config.root)

    source = extract_source(config)
    written = split_into_batches(config, source, store, reset_progress=False)
    for index in range(len(written)):
        store.mark_processed(index)

    source = extract_source(config)
    split_into_batches(config, source, store, reset_progress=False)

    assert store.last_processed == len(written) - 1
    assert len(store.history) == len(written)
    assert [entry["batch_idx"] for entry in store.history] == list(range(len(written)))


# ---------------------------------------------------------------------------
# Состояние
# ---------------------------------------------------------------------------


def test_dataset_hash_is_exposed(tmp_path):
    """Хэш данных доступен из состояния: по нему решается, те ли данные."""
    config, _ = make_project(tmp_path)
    store = StateStore(config.state_file, root=config.root)
    assert store.dataset_hash is None

    store.init_batches(["a.csv"], dataset_hash="abc")
    assert store.dataset_hash == "abc"


def test_missing_batches_detects_lost_files(tmp_path):
    """Потерянные файлы батчей обнаруживаются, а не считаются обработанными."""
    (tmp_path / "raw_batches").mkdir()
    (tmp_path / "raw_batches" / "batch_2018-01.csv").write_text("x", encoding="utf-8")

    store = StateStore(tmp_path / "state.json", root=tmp_path)
    store.init_batches(
        ["raw_batches/batch_2018-01.csv", "raw_batches/batch_2018-02.csv"]
    )

    assert store.missing_batches() == ["raw_batches/batch_2018-02.csv"]


def test_missing_batches_empty_when_all_present(tmp_path):
    """Все файлы на месте — список пропавших пуст."""
    (tmp_path / "raw_batches").mkdir()
    (tmp_path / "raw_batches" / "batch_2018-01.csv").write_text("x", encoding="utf-8")

    store = StateStore(tmp_path / "state.json", root=tmp_path)
    store.init_batches(["raw_batches/batch_2018-01.csv"])

    assert store.missing_batches() == []


def test_state_file_survives_reload(tmp_path):
    """Отметка о прогрессе переживает перечитывание файла."""
    store = StateStore(tmp_path / "state.json", root=tmp_path)
    store.init_batches(["a.csv", "b.csv"])
    store.mark_processed(0)

    reloaded = StateStore(tmp_path / "state.json", root=tmp_path)
    assert reloaded.last_processed == 0
    assert reloaded.has_next is True
    assert reloaded.remaining() == 1


def test_state_json_remains_valid_json(tmp_path):
    """Файл состояния остаётся читаемым: от него зависит весь CI."""
    store = StateStore(tmp_path / "state.json", root=tmp_path)
    store.init_batches(["a.csv", "b.csv"], dataset_hash="h")
    store.mark_processed(0, batch_detail={"f1": 0.2})

    data = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert data["batches"] == ["a.csv", "b.csv"]
    assert data["last_processed"] == 0
    assert data["dataset_hash"] == "h"


# ---------------------------------------------------------------------------
# Состав вызова в CI
# ---------------------------------------------------------------------------


def test_workflow_calls_init_without_reset():
    """В CI `init` вызывается из скрипта запуска, без `-reset`.

    Workflow зовёт `scripts/run_pipeline.sh`, а не `run.py` напрямую,
    поэтому проверяется именно вызов скрипта: если бы сброс просочился
    туда, каждое срабатывание расписания начинало бы обучение заново.
    """
    workflow = (
        Path(__file__).resolve().parent.parent
        / ".github" / "workflows" / "main.yml"
    ).read_text(encoding="utf-8")

    assert "run_pipeline.sh" in workflow
    assert "-reset" not in workflow


def test_script_calls_init_without_reset():
    """Скрипт запуска — тот же сценарий, что и в CI."""
    script = (
        Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.sh"
    ).read_text(encoding="utf-8")

    assert "-mode init" in script
    assert "-reset" not in script


@pytest.mark.parametrize(
    "flag, expected",
    [("-mode init -reset", True), ("-mode init --reset", True), ("-mode init", False)],
)
def test_reset_flag_is_understood(flag: str, expected: bool, capsys):
    """Флаг сброса разбирается и в краткой, и в длинной форме."""
    from run import build_parser

    args = build_parser().parse_args(flag.split())
    assert args.reset is expected
