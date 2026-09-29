"""Публикация дашборда истории обучения в вебе (7.b.i, задание 2 — 2.b.iii).

Требование: «сохранение dashboard истории обучения **или** запуск онлайн-
сервиса на платформе GitHub Actions». Здесь реализовано первое, причём
публикуется не просто файл, а собранный сайт: по требованию 7.b.i
дашборд должен быть сохранён, а не показан на экране разработчика, и
разворачивается он средствами самой платформы — GitHub Pages.

Почему генератор свой, а не готовый сервис (решение Р-4): статический
сайт собирается из артефактов, которые конвейер и так пишет, и не
требует ни браузера, ни сети, ни отдельного сервера. Логика проверяется
тестами, а результат — обычная папка файлов, которую понимает и Pages,
и любой другой хостинг.

Из чего состоит сайт:

* `index.html`      — сводка прогона и ссылки на всё остальное;
* `dashboard.html`  — сам дашборд истории обучения;
* `report.txt`      — последний текстовый отчёт;
* `summary.json`    — те же данные машинно-читаемо;
* `models/`         — продуктовая модель, препроцессор и сборщик данных;
* `meta.json`       — что именно опубликовано (машино-читаемо);
* `nojekyll`        — служебный файл. При публикации из ветки Jekyll
                      отбрасывает каталоги и файлы, начинающиеся с
                      подчёркивания; при развёртывании из Actions он
                      безвреден и снимает вопрос при смене источника.

Модуль запускается и как скрипт — так публикация проверяется в CI
отдельным шагом и локально:

    python -m src.publish --out site
    python -m src.publish --out site --json
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.config import Config, load_config
from src.dashboard import build_dashboard
from src.utils import save_json, to_json_serializable

logger = logging.getLogger(__name__)

#: Бюджет на публикуемые артефакты. На 48 батчей каталог `models/`
#: занимает десятки мегабайт: в сайт попадают только актуальные модели,
#: иначе публикация становится бессмысленно тяжёлой.
SIZE_BUDGET_BYTES = 20 * 1024 * 1024

#: Модели, которые имеет смысл публиковать. Имена — «последние»
#: экземпляры семейств, а не все версии: история версий видна в
#: дашборде и в реестре, в Site хватит работающей копии.
PUBLISHABLE_MODELS = (
    "best_model.pkl",
    "preprocessor.pkl",
    "collector.pkl",
    "lr_latest.pkl",
    "dt_latest.pkl",
    "rf_latest.pkl",
    "mlp_latest.pkl",
)


class PublishError(RuntimeError):
    """Сайт собрать не удалось."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SiteResult:
    """Что получилось опубликовать.

    Не только список файлов: счётчики нужны, чтобы шаг в CI мог
    сообщить, что публикация вообще что-то содержит, — иначе пустой
    сайт выглядел бы как успешная публикация.
    """

    root: Path
    files: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    dashboard: Path | None = None

    @property
    def total_bytes(self) -> int:
        return sum(self.files.values())

    @property
    def is_empty(self) -> bool:
        return not self.files

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "files": self.files,
            "total_bytes": self.total_bytes,
            "skipped": self.skipped,
            "has_dashboard": self.dashboard is not None,
        }


# ---------------------------------------------------------------------------
# Вспомогательные операции
# ---------------------------------------------------------------------------


def _latest(directory: Path, pattern: str) -> Path | None:
    """Самый свежий файл по шаблону.

    Сортировка по имени, а не по времени изменения: в CI время
    меняется у всех файлов разом, а имя содержит штамп в формате
    `YYYYmmdd_HHMMSS`, который по этому же правилу упорядочен лексикографически.
    """
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob(pattern))
    return candidates[-1] if candidates else None


def _reset_directory(root: Path) -> None:
    """Очистить каталог сайта.

    Старые файлы удаляются целиком: иначе при следующей публикации в
    `site/` остались бы отчёты и модели прошлых прогонов, и посетитель
    видел бы смешанную картину. Каталог удаляется только если это
    действительно наш каталог — путь задан конфигурацией, но лишняя
    проверка здесь ничего не стоит.
    """
    if not root.exists():
        root.mkdir(parents=True)
        return
    if root.is_file():
        raise PublishError(f"Путь сайта {root} занят файлом")
    for item in root.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)


def _copy(source: Path | None, target: Path, result: SiteResult) -> bool:
    """Скопировать файл, если он есть; иначе отметить как пропущенный.

    Returns:
        True, если файл опубликован.
    """
    if source is None or not source.is_file():
        label = source.name if source is not None else target.name
        logger.info("Пропущено (нет файла): %s", label)
        result.skipped.append(target.name)
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    result.files[target.name if target.parent == result.root
                 else f"{target.parent.name}/{target.name}"] = target.stat().st_size
    return True


# ---------------------------------------------------------------------------
# Сборка
# ---------------------------------------------------------------------------


def build_site(
    config: Config,
    output_dir: str | Path | None = None,
    *,
    refresh_dashboard: bool = True,
) -> SiteResult:
    """Собрать статический сайт с дашбордом истории обучения.

    Args:
        config: конфигурация конвейера.
        output_dir: куда собирать. По умолчанию `paths.site`.
        refresh_dashboard: пересобрать дашборд, даже если файл уже есть.
            Сборка дешевле повторного обучения, а набор метаданных мог
            измениться с прошлого раза.

    Returns:
        SiteResult со списком опубликованных файлов.

    Raises:
        PublishError: каталог сайта недоступен для записи.
    """
    root = Path(output_dir) if output_dir else config.path("site")
    try:
        _reset_directory(root)
    except OSError as error:
        raise PublishError(f"Не удалось подготовить каталог {root}: {error}") from error

    result = SiteResult(root=root)

    # 1. Дашборд истории обучения — центральная страница сайта.
    #    Пустой дашборд не публикуется: без манифестов прогонов он
    #    содержал бы одни пустые секции и выглядел бы как успешная
    #    публикация истории обучения, которой не было.
    manifests_present = any(config.metadata_dir.glob("run_manifest_*.json"))
    dashboard_src = config.reports_dir / "dashboard.html"
    if not manifests_present:
        if dashboard_src.is_file() and not refresh_dashboard:
            # Файл остался от прошлого прогона, а метаданных уже нет:
            # публиковать его нельзя, он показывает чужие данные.
            logger.info("Дашборд не публикуется: нет манифестов прогонов")
        result.skipped.append("dashboard.html")
    else:
        if refresh_dashboard or not dashboard_src.is_file():
            try:
                dashboard_src = build_dashboard(
                    config.reports_dir, config.metadata_dir
                )
            except Exception as error:  # noqa: BLE001
                raise PublishError(f"Не удалось собрать дашборд: {error}") from error

        if _copy(dashboard_src, root / "dashboard.html", result):
            result.dashboard = root / "dashboard.html"

    # 2. Отчёты: последние, а не все — полная история остаётся в артефактах.
    latest_txt = _latest(config.reports_dir, "summary_*.txt")
    _copy(latest_txt, root / "report.txt", result)
    latest_json = _latest(config.reports_dir, "summary_*.json")
    _copy(latest_json, root / "summary.json", result)

    # 3. Модели и сборщик — с бюджетом по размеру.
    budget = SIZE_BUDGET_BYTES
    for name in PUBLISHABLE_MODELS:
        source = config.models_dir / name
        if not source.is_file():
            result.skipped.append(name)
            continue
        size = source.stat().st_size
        if size > budget:
            logger.info(
                "Пропущено (превышен бюджет %d МБ): %s (%.1f МБ)",
                budget // (1024 * 1024), name, size / (1024 * 1024),
            )
            result.skipped.append(name)
            continue
        _copy(source, root / "models" / name, result)
        budget -= size

    # 4. Служебные файлы.
    (root / "nojekyll").write_text("", encoding="utf-8")
    result.files["nojekyll"] = 0

    # 5. `meta.json` **не перечисляет файлы поимённо**. Такой список
    #    не может быть верен: записать его можно только до того, как
    #    известен собственный размер файла, и до того, как существует
    #    визитка. Перечисление живёт в `index.html`, где оно и нужно
    #    человеку, а здесь — только счётчики и признаки, которые можно
    #    назвать честно.
    base = {
        "built_at": _utc_now(),
        "config_hash": config.config_hash,
        "batches_total": len(_state_batches(config)),
        "source_zip": str(config.paths.get("source_zip", "")),
    }
    save_json(root / "meta.json", {
        **base,
        # Считаются и сам `meta.json`, и визитка: на момент записи их
        # ещё нет в инвентаре, но в каталоге они будут.
        "n_published_files": len(result.files) + 2,
        "n_published_bytes": result.total_bytes,
        "has_dashboard": result.dashboard is not None,
        "skipped": sorted(set(result.skipped)),
    })
    result.files["meta.json"] = (root / "meta.json").stat().st_size

    # 6. Визитка — последняя: к этому моменту известны все файлы, кроме
    #    неё самой, и её размер попадает в таблицу на странице.
    index = _render_index(config, result, {**base, **_inventory(result)})
    (root / "index.html").write_text(index, encoding="utf-8")
    result.files["index.html"] = (root / "index.html").stat().st_size

    logger.info(
        "Сайт собран: %s (%d файлов, %.1f КБ)",
        root, len(result.files), result.total_bytes / 1024,
    )
    return result


def _inventory(result: SiteResult) -> dict[str, Any]:
    """Состав публикации на текущий момент."""
    return {
        "files": dict(result.files),
        "total_bytes": result.total_bytes,
        "skipped": sorted(set(result.skipped)),
    }


def _state_batches(config: Config) -> list[str]:
    """Список батчей из состояния; пустой список, если его ещё нет.

    Сайт должен собираться и до первого обучения — иначе первый же
    сбой в конвейере означал бы, что опубликовать нечего, и зелёный
    статус прогона вводил бы в заблуждение.
    """
    try:
        from src.state import StateStore

        return list(StateStore(config.state_file, root=config.root).batches)
    except Exception as error:  # noqa: BLE001
        logger.warning("Состояние недоступно (%s), сайт собирается без него", error)
        return []


# ---------------------------------------------------------------------------
# Отрисовка
# ---------------------------------------------------------------------------

_STYLES = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem 4rem;
  font: 16px/1.55 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  color: #111827; background: #f9fafb;
}
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.1rem; margin: 2rem 0 .75rem; }
.sub { color: #6b7280; margin: 0 0 1.5rem; font-size: .92rem; }
.cards { display: grid; gap: .75rem; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); }
.card { background: #fff; border: 1px solid #e5e7eb; border-radius: .6rem; padding: .85rem 1rem; }
.card .k { color: #6b7280; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }
.card .v { font-size: 1.3rem; font-weight: 600; margin-top: .2rem; word-break: break-all; }
table { border-collapse: collapse; width: 100%; background: #fff; font-size: .9rem; }
th, td { border: 1px solid #e5e7eb; padding: .45rem .6rem; text-align: left; }
th { background: #f3f4f6; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
a { color: #2563eb; }
.cta { display: inline-block; background: #2563eb; color: #fff; text-decoration: none;
       padding: .6rem 1.1rem; border-radius: .5rem; font-weight: 600; }
.cta:hover { background: #1d4ed8; }
.note { color: #6b7280; font-size: .85rem; }
@media (prefers-color-scheme: dark) {
  body { background: #0b1220; color: #e5e7eb; }
  .card, table { background: #111a2b; border-color: #24314a; }
  th { background: #18233a; }
  a, .cta { color: #60a5fa; }
  .cta { color: #0b1220; }
}
"""


def _human(size: float) -> str:
    for unit in ("Б", "КБ", "МБ"):
        if size < 1024 or unit == "МБ":
            return f"{size:.0f} {unit}" if unit == "Б" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} МБ"


#: Служебные файлы сайта. В списке для скачивания они не нужны: это
#: либо сама визитка, либо маркер для хостинга.
SERVICE_FILES = ("index.html", "nojekyll")


def _when(iso: str) -> str:
    """Метка времени в виде, пригодном для чтения человеком.

    В `meta.json` остаётся ISO — там формат важен для машин, а на
    странице метка вида `2026-09-29T20:17:46+00:00` рвётся посреди
    строки и ни о чём не говорит.
    """
    try:
        moment = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return str(iso)
    return moment.strftime("%d.%m.%Y %H:%M UTC")


def _render_index(
    config: Config, result: SiteResult, meta: dict[str, Any]
) -> str:
    """Страница-визитка со сводкой и ссылками на артефакты.

    Вызывается последней, когда известны все файлы публикации, кроме
    самой визитки. Поэтому в счётчике она учитывается отдельно, а в
    таблице её нет — вместо неё указано, сколько файлов служебных.
    """
    escape = html.escape
    manifest = _latest(config.metadata_dir, "run_manifest_*.json")
    production = "—"
    batches_done = "—"
    if manifest is not None:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            registry = data.get("registry") or {}
            current = registry.get("production") or {}
            if current:
                production = "{model} {version}".format(
                    model=current.get("model", "?"),
                    version=current.get("version", "?"),
                )
            # `batch_idx` — индекс последнего обработанного батча с нуля,
            # поэтому обработано его значение плюс один.
            index = data.get("batch_idx")
            if isinstance(index, int):
                batches_done = str(index + 1)
        except (OSError, ValueError, TypeError) as error:
            logger.warning("Манифест %s не разобран: %s", manifest.name, error)

    # Визитка записывается после сборки страницы, поэтому в счётчик
    # входит отдельно: иначе он показывал бы на один меньше, чем
    # файлов в каталоге, и расхождение было бы видно посетителю.
    published = len(result.files) + 1
    cards = [
        ("Батчей обработано", batches_done),
        ("Продуктовая модель", production),
        ("Хэш конфигурации", meta["config_hash"][:12]),
        ("Опубликовано файлов", str(published)),
        ("Объём сайта", _human(float(meta.get("total_bytes", 0)))),
        ("Сборка", _when(meta["built_at"])),
    ]
    cards_html = "\n".join(
        f'<div class="card"><div class="k">{escape(key)}</div>'
        f'<div class="v">{escape(value)}</div></div>'
        for key, value in cards
    )

    rows = "\n".join(
        f'<tr><td><a href="{escape(name)}">{escape(name)}</a></td>'
        f'<td class="num">{_human(size)}</td></tr>'
        for name, size in sorted(result.files.items())
        if name not in SERVICE_FILES
    )

    dashboard_link = (
        '<a class="cta" href="dashboard.html">Открыть дашборд истории обучения</a>'
        if result.dashboard is not None
        else '<span class="note">Дашборд не собран: нет метаданных прогонов.</span>'
    )

    skipped = ""
    if result.skipped:
        skipped = (
            '<h2>Не опубликовано</h2><p class="note">'
            f'Отсутствовали или превысили бюджет размера: '
            f'{escape(", ".join(sorted(set(result.skipped))))}.</p>'
        )

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLOps-конвейер — дашборд истории обучения</title>
<style>{_STYLES}</style>
</head>
<body>
<main>
  <h1>MLOps-конвейер обработки потоковых табличных данных</h1>
  <p class="sub">
    Страница собрана автоматически на GitHub Actions после каждого прогона
    обучения. Здесь публикуются история обучения, отчёты и актуальные
    модели; полные логи остаются в артефактах запуска.
  </p>

  <p>{dashboard_link}</p>

  <h2>Сводка прогона</h2>
  <div class="cards">
{cards_html}
  </div>

  <h2>Опубликованные файлы</h2>
  <table>
    <tr><th>Файл</th><th style="text-align:right">Размер</th></tr>
{rows}
  </table>
  <p class="note">
    Служебные файлы сайта (<code>index.html</code>, <code>nojekyll</code>)
    в список не выведены: они нужны хостингу, а не посетителю.
    Полный состав публикации — в <a href="meta.json">meta.json</a>.
  </p>

  {skipped}

  <h2>Как это работает</h2>
  <p class="note">
    1. Загрузка источника и разбиение на помесячные батчи
    (<code>-mode init</code>), сохранение сериализуемого сборщика данных.<br>
    2. Дообучение на очередном батче: очистка, ассоциативные правила,
    признаки, четыре модели, дрейф, гейт качества
    (<code>-mode update</code>).<br>
    3. Отчёт и дашборд по истории прогонов (<code>-mode summary</code>),
    публикация сайта (<code>-mode publish</code>).<br>
    4. Приём новых данных извне:
    <code>python -m src.collector --append новый_батч.csv</code>.
  </p>
</main>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Командный интерфейс
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.publish",
        description="Собрать статический сайт с дашбордом истории обучения",
    )
    parser.add_argument(
        "--out", metavar="DIR", default=None,
        help="Каталог сайта (по умолчанию — paths.site из конфигурации)",
    )
    parser.add_argument(
        "--config", metavar="FILE", default=None,
        help="Путь к конфигурационному файлу",
    )
    parser.add_argument(
        "--keep-dashboard", action="store_true",
        help="Не пересобирать дашборд, если файл уже существует",
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

    try:
        config = load_config(args.config)
        result = build_site(
            config,
            args.out,
            refresh_dashboard=not args.keep_dashboard,
        )
    except Exception as error:  # noqa: BLE001
        if args.json:
            print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        else:
            print(f"Ошибка публикации: {error}", file=sys.stderr)
        return 1

    payload = to_json_serializable({"ok": True, **result.as_dict()})
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Сайт: {result.root}")
        for name, size in sorted(result.files.items()):
            print(f"  {name:<28} {_human(size)}")
        if result.skipped:
            print(f"Не опубликовано: {', '.join(sorted(set(result.skipped)))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
