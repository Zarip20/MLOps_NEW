"""Состояние прогона (АР-4).

`state.json` хранит список батчей, номер последнего обработанного и историю
прогонов. Ключевые свойства:

* **Версионирование схемы.** Поле `schema_version` позволяет отказаться
  читать состояние, записанное несовместимой версией кода.
* **Атомарность.** Запись идёт через временный файл с последующим
  переименованием, поэтому оборванный процесс не оставляет
  наполовину записанный `state.json`.
* **Продвижение только при успехе.** Батч отмечается обработанным лишь
  после того, как конвейер завершился без ошибок (АР-11).

Состояние — единственное, что позволяет инкрементальному режиму
продолжить обработку с места обрыва, в том числе между запусками CI.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 2

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StateStore:
    """Типобезопасный доступ к состоянию прогона."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        root: str | os.PathLike[str] | None = None,
    ) -> None:
        """Args:
            path: файл состояния.
            root: корень проекта, относительно которого записаны пути к
                батчам. По умолчанию — каталог самого файла состояния,
                что верно для штатного расположения.
        """
        self.path = Path(path)
        self.root = Path(root) if root is not None else self.path.parent
        self._state = self._read()

    # -- чтение ----------------------------------------------------------

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "dataset_hash": None,
            "config_hash": None,
            "batches": [],
            "last_processed": -1,
            "history": [],
            "runs": [],
        }

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self._empty()
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError) as error:
            raise RuntimeError(
                f"Не удалось прочитать состояние {self.path}: {error}. "
                f"Удалите файл, чтобы начать обработку заново."
            ) from error

        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise RuntimeError(
                f"Схема состояния {self.path} не совпадает: "
                f"на диске версия {version}, ожидалась {SCHEMA_VERSION}. "
                f"Перезапустите инициализацию."
            )
        return data

    # -- доступ к полям --------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._state[key]

    @property
    def batches(self) -> list[str]:
        return list(self._state.get("batches", []))

    @property
    def last_processed(self) -> int:
        return int(self._state.get("last_processed", -1))

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._state.get("history", []))

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[str]:
        return iter(self.batches)

    @property
    def is_initialised(self) -> bool:
        return bool(self.batches)

    @property
    def dataset_hash(self) -> str | None:
        """Хэш исходных данных, зафиксированный при инициализации."""
        return self._state.get("dataset_hash")

    def missing_batches(self) -> list[str]:
        """Батчи из состояния, файлов которых нет на диске.

        Нужно, чтобы отличить «уже разбито» от «файлы потеряны»: в CI
        каталог `data/raw_batches/` не сохраняется в кэше, поэтому на
        каждом запуске батчи приходится создавать заново, и состояние
        при этом остаётся верным.
        """
        return [path for path in self.batches if not (self.root / path).is_file()]

    @property
    def has_next(self) -> bool:
        return self.last_processed + 1 < len(self.batches)

    def remaining(self) -> int:
        return max(len(self.batches) - self.last_processed - 1, 0)

    def next_batch(self) -> tuple[str, int] | None:
        """Путь и индекс следующего необработанного батча."""
        index = self.last_processed + 1
        if index >= len(self.batches):
            return None
        return self.batches[index], index

    # -- запись ----------------------------------------------------------

    def save(self) -> None:
        """Атомарно записать состояние на диск."""
        self._state["updated_at"] = _utc_now()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Временный файл в том же каталоге: rename в пределах тома атомарен.
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self.path.parent),
            prefix=".state-",
            suffix=".tmp",
            delete=False,
        )
        try:
            with handle:
                json.dump(self._state, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, self.path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise

    def init_batches(
        self,
        batches: list[str],
        dataset_hash: str | None = None,
        reset: bool = True,
    ) -> bool:
        """Зафиксировать список батчей.

        Args:
            batches: относительные пути к батчам.
            dataset_hash: хэш исходных данных.
            reset: сбрасывать ли прогресс, даже если список батчей не
                изменился.

        Returns:
            True, если прогресс был сброшен.

        Список батчей может совпасть при разных обстоятельствах: данные
        не изменились, либо `init` вызван повторно в том же прогоне.
        Сбрасывать в этом случае прогресс нельзя — отметка о батчах
        остаётся верной, а инкрементальный сценарий (дообучение в CI от
        состояния прошлого запуска) без этого превращается в обучение с
        нуля. Если же набор данных действительно другой, хэш не совпадёт
        и вызывающий код запросит сброс явно.
        """
        new_list = [str(path) for path in batches]
        same_list = new_list == list(self._state.get("batches", []))

        if same_list and not reset:
            if dataset_hash:
                self._state["dataset_hash"] = dataset_hash
            self.save()
            logger.info(
                "Список батчей не изменился (%d), прогресс сохранён: "
                "обработано %d из %d",
                len(new_list), self.last_processed + 1, len(new_list),
            )
            return False

        if same_list and self.last_processed >= 0:
            logger.warning(
                "Список батчей прежний, но прогресс затребован к сбросу: "
                "будет выполнено обучение с нуля"
            )

        self._state.update(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": _utc_now(),
                "dataset_hash": dataset_hash,
                "batches": new_list,
                "last_processed": -1,
                "history": [],
            }
        )
        self.save()
        return True

    def add_batches(
        self, batches: list[str], dataset_hash: str | None = None
    ) -> int:
        """Дописать новые батчи в конец списка, не сдвигая прогресс.

        Используется синхронизацией с каталогом: батчи, принятые сборщиком
        извне, должны обрабатываться после уже обработанных, иначе
        индексы в истории перестанут соответствовать файлам.
        """
        if not batches:
            return 0
        self._state["batches"] = list(self._state.get("batches", [])) + list(batches)
        if dataset_hash:
            self._state["dataset_hash"] = dataset_hash
        self.save()
        return len(batches)

    def mark_processed(
        self,
        index: int,
        *,
        dataset_hash: str | None = None,
        config_hash: str | None = None,
        **details: Any,
    ) -> None:
        """Отметить батч обработанным и записать сведения о запуске."""
        if index < 0 or index >= len(self.batches):
            raise IndexError(
                f"Индекс батча {index} вне диапазона [0, {len(self.batches) - 1}]"
            )
        if index <= self.last_processed:
            return

        entry = {
            "batch_idx": index,
            "batch": self.batches[index],
            "timestamp": _utc_now(),
            # Хэши дублируются в записи истории, а не только в корне
            # состояния: по истории восстанавливается, на какой версии
            # конфигурации и каких данных обучалась модель этого батча.
            **({"dataset_hash": dataset_hash} if dataset_hash else {}),
            **({"config_hash": config_hash} if config_hash else {}),
            **details,
        }
        self._state["last_processed"] = index
        self._state.setdefault("history", []).append(entry)
        if dataset_hash:
            self._state["dataset_hash"] = dataset_hash
        if config_hash:
            self._state["config_hash"] = config_hash
        self.save()

    def record_run(self, **details: Any) -> None:
        """Записать сводку о запуске (используется в манифесте и отчёте)."""
        self._state.setdefault("runs", []).append(
            {"timestamp": _utc_now(), **details}
        )
        self.save()

    def reset(self) -> None:
        """Сбросить прогресс, сохранив список батчей."""
        self._state["last_processed"] = -1
        self._state["history"] = []
        self.save()
