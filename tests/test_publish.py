"""Проверки публикации дашборда истории обучения (задание 2, пункт 2.b.iii).

Публикация — это не «скопировать файл», а сборка сайта, который затем
разворачивается на GitHub Pages. Проверяются свойства, из-за которых
такой сайт перестаёт быть полезным: что старые файлы не копятся, что
пустой прогон не выглядит успешной публикацией, что ссылки на странице
ведут на файлы, которые действительно есть.
"""

from __future__ import annotations

import copy
import json
import re

import pytest

from src.config import Config, load_config
from src.publish import (
    PUBLISHABLE_MODELS,
    SERVICE_FILES,
    PublishError,
    SiteResult,
    build_site,
    main,
)


# ---------------------------------------------------------------------------
# Подготовка
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path):
    """Боевая конфигурация с артефактами во временном каталоге.

    Пути переносятся в `tmp_path`, чтобы тест не зависел от накопленного
    состояния проекта и не портил его. `Config` неизменяемый, поэтому
    собирается новый объект, а не правка существующего.
    """
    real = load_config()
    data = copy.deepcopy(real.data)
    data["paths"] = {
        "data_raw": "raw_batches",
        "data_processed": "processed",
        "models": "models",
        "reports": "reports",
        "metadata": "metadata",
        "rules_file": "rules.json",
        "state_file": "state.json",
        "site": "site",
    }
    prepared = Config(data=data, root=tmp_path)
    prepared.ensure_dirs()
    return prepared


def write_dashboard(config) -> None:
    """Разложить результат прогона: дашборд и манифест.

    Дашборд публикуется только при наличии манифестов, поэтому почти
    каждой проверке публикации нужны оба файла. Случай «дашборд есть,
    а манифестов нет» разбирается отдельно — через `stale_dashboard`.
    """
    (config.reports_dir / "dashboard.html").write_text(
        "<!DOCTYPE html><html><body>дашборд</body></html>", encoding="utf-8"
    )
    write_manifest(config)


def stale_dashboard(config) -> None:
    """Дашборд без манифестов — артефакт прошлого прогона."""
    (config.reports_dir / "dashboard.html").write_text(
        "<!DOCTYPE html><html><body>прошлый дашборд</body></html>", encoding="utf-8"
    )


def write_reports(config) -> None:
    (config.reports_dir / "summary_20240101_000000.txt").write_text(
        "старый отчёт", encoding="utf-8"
    )
    (config.reports_dir / "summary_20240202_121212.txt").write_text(
        "новый отчёт", encoding="utf-8"
    )
    (config.reports_dir / "summary_20240202_121212.json").write_text(
        '{"batches_processed": 5}', encoding="utf-8"
    )


def write_models(config) -> None:
    for name in PUBLISHABLE_MODELS:
        (config.models_dir / name).write_bytes(b"model")


def write_manifest(
    config, *, model: str = "lr", version: str = "v0001", batch_idx: int = 0
) -> None:
    """Манифест прогона в том же формате, что пишет конвейер.

    Именно он служит источником данных для визитки: без манифестов
    дашборд не публикуется, поэтому проверки публикации обязаны их
    создавать.
    """
    (config.metadata_dir / "run_manifest_0000.json").write_text(
        json.dumps(
            {
                "batch_idx": batch_idx,
                "registry": {
                    "production": {
                        "version": version,
                        "model": model,
                        "batch_idx": batch_idx,
                        "f1": 0.25,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Сборка
# ---------------------------------------------------------------------------


def test_site_contains_dashboard_and_index(config, tmp_path):
    """Сайт состоит из визитки и самого дашборда."""
    write_dashboard(config)
    result = build_site(config, refresh_dashboard=False)

    assert (tmp_path / "site" / "index.html").is_file()
    assert (tmp_path / "site" / "dashboard.html").is_file()
    assert result.dashboard is not None
    assert not result.is_empty


def test_dashboard_is_copied_verbatim(config, tmp_path):
    """Содержимое дашборда не переписывается при публикации."""
    write_dashboard(config)
    build_site(config, refresh_dashboard=False)
    assert (tmp_path / "site" / "dashboard.html").read_text(
        encoding="utf-8"
    ).endswith("дашборд</body></html>")


def test_only_latest_reports_are_published(config, tmp_path):
    """Публикуется последний отчёт, а не вся история прогонов.

    Иначе со временем каталог разрастался бы, а посетитель сайта
    не смог бы понять, какой прогон актуален.
    """
    write_dashboard(config)
    write_reports(config)
    build_site(config, refresh_dashboard=False)

    site = tmp_path / "site"
    assert site.joinpath("report.txt").read_text(encoding="utf-8") == "новый отчёт"
    payload = json.loads(site.joinpath("summary.json").read_text(encoding="utf-8"))
    assert payload["batches_processed"] == 5


def test_models_are_published_under_models_dir(config, tmp_path):
    """Модели и сборщик публикуются в отдельном каталоге."""
    write_dashboard(config)
    write_models(config)
    result = build_site(config, refresh_dashboard=False)

    for name in PUBLISHABLE_MODELS:
        assert (tmp_path / "site" / "models" / name).is_file()
    assert "models/collector.pkl" in result.files


def test_nojekyll_marker_is_written(config, tmp_path):
    """Служебный файл `nojekyll` создаётся.

    При публикации из ветки Jekyll отбрасывает каталоги и файлы с
    подчёркиванием в имени; при развёртывании из Actions файл
    безвреден, но снимает вопрос, если источник Pages сменят.
    """
    write_dashboard(config)
    build_site(config, refresh_dashboard=False)
    assert (tmp_path / "site" / "nojekyll").is_file()


def test_meta_json_describes_the_build(config, tmp_path):
    """В meta.json — сведения о сборке, без перечисления файлов.

    Перечисление поимённо не может быть верен: файл записывается раньше,
    чем известны его собственный размер и визитка. Поэтому здесь только
    счётчики, которые можно назвать честно, а состав — на визитке.
    """
    write_dashboard(config)
    result = build_site(config, refresh_dashboard=False)

    meta = json.loads((tmp_path / "site" / "meta.json").read_text(encoding="utf-8"))
    assert meta["config_hash"] == config.config_hash
    assert meta["built_at"].startswith("20")
    assert meta["has_dashboard"] is True
    assert meta["n_published_files"] == len(result.files)
    assert "files" not in meta, "поимённый список в meta.json не может быть верен"


def test_index_count_matches_files_on_disk(config, tmp_path):
    """Счётчик файлов на визитке совпадает с тем, что лежит в каталоге.

    Визитка пишется последней и в момент своей сборки ещё не учтена в
    инвентаре, поэтому счётчик без поправки показывал бы на единицу
    меньше — расхождение сразу видно посетителю.
    """
    write_dashboard(config)
    write_models(config)
    result = build_site(config, refresh_dashboard=False)

    on_disk = sum(1 for p in (tmp_path / "site").rglob("*") if p.is_file())
    assert on_disk == len(result.files)

    page = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert f">{on_disk}</div>" in page, "счётчик файлов на странице неверен"


def test_index_time_is_readable(config, tmp_path):
    """Метка времени читаема: ISO-строка посреди HTML рвётся и не говорит ни о чём."""
    write_dashboard(config)
    build_site(config, refresh_dashboard=False)

    page = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert "T20:" not in page and "T21:" not in page
    assert re.search(r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2} UTC", page)

    # При этом в meta.json формат остаётся машинным
    meta = json.loads((tmp_path / "site" / "meta.json").read_text(encoding="utf-8"))
    assert meta["built_at"].startswith("20") and "T" in meta["built_at"]


def test_table_covers_every_published_file(config, tmp_path):
    """В таблице есть всё, кроме служебных файлов.

    Расхождение между числом файлов в каталоге и числом строк в таблице
    бросается в глаза: посетитель видит «13 файлов» и одиннадцать строк
    и не понимает, куда делись остальные.
    """
    write_dashboard(config)
    write_reports(config)
    write_models(config)
    result = build_site(config, refresh_dashboard=False)

    page = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    links = set(re.findall(r'href="([^"#]+)"', page))
    expected = set(result.files) - set(SERVICE_FILES) - {"index.html"}
    assert expected <= links, f"в таблице нет: {expected - links}"
    assert links & {"dashboard.html", "report.txt", "summary.json", "meta.json"}


def test_service_files_are_not_offered_for_download(config, tmp_path):
    """Служебные файлы не попадают в список для скачивания."""
    write_dashboard(config)
    write_models(config)
    build_site(config, refresh_dashboard=False)

    page = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    links = re.findall(r'href="([^"#]+)"', page)
    assert "nojekyll" not in links
    assert "index.html" not in links
    # Остальные на месте
    assert "dashboard.html" in links
    assert "models/collector.pkl" in links


# ---------------------------------------------------------------------------
# Очистка между прогонами
# ---------------------------------------------------------------------------


def test_stale_files_are_removed_on_rebuild(config, tmp_path):
    """Сборка заново не оставляет файлов прошлого прогона.

    Иначе на сайте одновременно окажутся модели и отчёты разных прогонов,
    и это выглядело бы как результат последнего обучения.
    """
    write_dashboard(config)
    write_models(config)
    build_site(config, refresh_dashboard=False)
    stale = tmp_path / "site" / "models" / "obsolete.pkl"
    stale.write_bytes(b"stale")
    (tmp_path / "site" / "index.html").write_text("прежняя визитка", encoding="utf-8")

    result = build_site(config, refresh_dashboard=False)

    assert not stale.exists()
    assert "obsolete.pkl" not in result.files
    assert (tmp_path / "site" / "index.html").read_text(encoding="utf-8") != "прежняя визитка"


def test_rebuild_after_model_removed_does_not_claim_it(config, tmp_path):
    """Пропавший артефакт попадает в skipped, а не выдаётся за опубликованный."""
    write_dashboard(config)
    write_models(config)
    build_site(config, refresh_dashboard=False)
    (config.models_dir / "collector.pkl").unlink()

    result = build_site(config, refresh_dashboard=False)

    assert "collector.pkl" in result.skipped
    assert "models/collector.pkl" not in result.files


# ---------------------------------------------------------------------------
# Пустой прогон
# ---------------------------------------------------------------------------


def test_site_builds_without_dashboard(config, tmp_path):
    """Сайт собирается и без дашборда — и честно об этом сообщает."""
    result = build_site(config, refresh_dashboard=True)

    assert (tmp_path / "site" / "index.html").is_file()
    assert result.dashboard is None
    assert "dashboard.html" in result.skipped
    assert not result.is_empty


def test_stale_dashboard_is_not_published(config, tmp_path):
    """Дашборд прошлого прогона не публикуется без манифестов.

    Иначе на сайте висел бы график обучения, которого не было в этом
    прогоне, и визитка выглядела бы актуальнее данных.
    """
    stale_dashboard(config)
    result = build_site(config, refresh_dashboard=False)

    assert result.dashboard is None
    assert not (tmp_path / "site" / "dashboard.html").exists()


def test_index_links_point_to_existing_files(config, tmp_path):
    """Каждая ссылка на визитке ведёт на файл, который действительно есть.

    Ссылка на отсутствующий файл — самая частая поломка статического
    сайта: страница выглядит готовой, а открывать нечего.
    """
    import re

    write_dashboard(config)
    write_reports(config)
    write_models(config)
    result = build_site(config, refresh_dashboard=False)

    page = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    targets = set(re.findall(r'href="([^"#]+)"', page))
    assert targets
    for target in targets:
        assert (tmp_path / "site" / target).is_file(), f"битая ссылка: {target}"


def test_index_escapes_injected_text(config, tmp_path):
    """Значения из артефактов не могут разметить страницу.

    На визитку попадают имя продуктовой модели и имена файлов из
    манифеста и каталогов; без экранирования строка вида `<script>`
    из метаданных выполнилась бы в браузере посетителя.
    """
    write_dashboard(config)
    write_manifest(config, model='<script>alert("x")</script>')
    build_site(config, refresh_dashboard=False)
    page = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page


def test_index_survives_broken_manifest(config, tmp_path):
    """Битый манифест не роняет публикацию: визитка собирается всегда.

    Метаданные пишутся в самом конце обучения, поэтому оборванный
    файл вполне возможен — и он не должен превращать успешный прогон
    в упавший шаг публикации.
    """
    write_dashboard(config)
    (config.metadata_dir / "run_manifest_0000.json").write_text(
        "{ обрезано", encoding="utf-8"
    )

    result = build_site(config, refresh_dashboard=False)

    assert (tmp_path / "site" / "index.html").is_file()
    assert not result.is_empty


def test_index_shows_production_model_from_manifest(config, tmp_path):
    """Продуктовая модель берётся из реестра, а не выдумывается."""
    write_dashboard(config)
    write_manifest(config, model="mlp", version="v0140", batch_idx=47)
    build_site(config, refresh_dashboard=False)

    page = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert "mlp v0140" in page
    assert "48" in page


# ---------------------------------------------------------------------------
# Бюджет размера и ошибки
# ---------------------------------------------------------------------------


def test_oversized_model_is_skipped_within_budget(config, tmp_path, monkeypatch):
    """Модель больше бюджета не публикуется, остальные — публикуются."""
    import src.publish as publish

    monkeypatch.setattr(publish, "SIZE_BUDGET_BYTES", 64)

    write_dashboard(config)
    write_models(config)
    (config.models_dir / "mlp_latest.pkl").write_bytes(b"x" * 512)

    result = build_site(config, refresh_dashboard=False)

    assert "mlp_latest.pkl" in result.skipped
    assert "models/mlp_latest.pkl" not in result.files
    assert "models/lr_latest.pkl" in result.files


def test_path_taken_by_file_is_reported(config, tmp_path):
    """Каталог сайта, занятый файлом, даёт понятную ошибку, а не traceback."""
    write_dashboard(config)
    (tmp_path / "site").write_text("это файл", encoding="utf-8")

    with pytest.raises(PublishError) as error:
        build_site(config, refresh_dashboard=False)
    assert "файлом" in str(error.value)


# ---------------------------------------------------------------------------
# Командный интерфейс
# ---------------------------------------------------------------------------


def test_cli_reports_success(config, tmp_path, capsys, monkeypatch):
    """CLI собирает сайт и печатает состав публикации."""
    import src.publish as publish

    monkeypatch.setattr(
        publish, "load_config", lambda *args, **kwargs: config
    )
    monkeypatch.setattr(
        publish, "build_dashboard",
        lambda reports, metadata, output=None: (
            (config.reports_dir / "dashboard.html")
        ),
    )
    write_dashboard(config)
    write_models(config)

    code = main(["--out", str(tmp_path / "published"), "--keep-dashboard"])

    assert code == 0
    out = capsys.readouterr().out
    assert "index.html" in out
    assert (tmp_path / "published" / "index.html").is_file()


def test_cli_returns_error_code_on_failure(config, capsys, monkeypatch):
    """Сбой сборки дашборда даёт ненулевой код возврата, а не тишину."""
    import src.publish as publish

    monkeypatch.setattr(
        publish, "load_config", lambda *args, **kwargs: config
    )
    monkeypatch.setattr(
        publish, "build_dashboard",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("нет данных")),
    )
    # Без манифестов сборщик к дашборду и не обращается, поэтому
    # сбой нужно воспроизвести на прогоне, который их имеет.
    write_manifest(config)

    code = main(["--out", "site"])

    assert code == 1
    assert "Ошибка публикации" in capsys.readouterr().err


def test_site_result_reports_totals():
    """SiteResult сам считает объём и пустоту — на них опирается CI."""
    empty = SiteResult(root="site")
    assert empty.is_empty
    assert empty.total_bytes == 0

    filled = SiteResult(root="site", files={"a.html": 10, "models/b.pkl": 5})
    assert not filled.is_empty
    assert filled.total_bytes == 15
