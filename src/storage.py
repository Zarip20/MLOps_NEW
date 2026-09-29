"""Накопительное хранилище обучающих данных (АР-8).

Каждая очищенная train-часть батча сохраняется отдельным сжатым файлом
`train_<idx>.csv.gz`. Это даёт три свойства, которых не было в исходной
реализации:

* **Состояние переживает перезапуск.** Пайплайн можно остановить и продолжить
  с того же места, в том числе в CI между запусками.
* **Очистка воспроизводима.** Ставка «дерево переобучается с нуля на всех
  накопленных данных» из `task.md` §4.1 требует именно такого буфера; в
  исходном коде его не было, поэтому дерево обучалось на одном батче (D-3).
* **Прослеживаемость.** По именам файлов видно, какие батчи участвовали
  в обучении, а `stats()` отдаёт эти сведения в метаданные.

В хранилище попадает **только train-часть** батча. Val-часть не
сохраняется никогда — иначе получился бы тот самый случай, когда
оценка производится на данных, участвовавших в обучении (D-2).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

_PART_PATTERN = re.compile(r"^train_(\d+)\.csv\.gz$")
_INDEX_NAME = "_index.json"


class TrainingStore:
    """Каталог с накопленными обучающими частями батчей."""

    def __init__(self, path: str | Path, columns: Sequence[str]) -> None:
        self.path = Path(path)
        self.columns = list(columns)
        self.path.mkdir(parents=True, exist_ok=True)

    # -- чтение ----------------------------------------------------------

    def _parts(self) -> list[tuple[int, Path]]:
        found: list[tuple[int, Path]] = []
        for item in self.path.glob("train_*.csv.gz"):
            match = _PART_PATTERN.match(item.name)
            if match:
                found.append((int(match.group(1)), item))
        return sorted(found)

    def __len__(self) -> int:
        return len(self._parts())

    def batch_indices(self) -> list[int]:
        return [index for index, _ in self._parts()]

    def load(self, window: int | None = None) -> pd.DataFrame:
        """Прочитать накопленные данные.

        Args:
            window: если задано, читать только последние `window` батчей.
                По умолчанию — вся история.
        """
        parts = self._parts()
        if window is not None and window > 0:
            parts = parts[-window:]
        if not parts:
            return pd.DataFrame(columns=self.columns)

        frames = [pd.read_csv(item) for _, item in parts]
        combined = pd.concat(frames, ignore_index=True)
        return combined

    # -- запись ----------------------------------------------------------

    def append(self, batch_idx: int, frame: pd.DataFrame) -> int:
        """Дописать train-часть батча. Возвращает число сохранённых строк.

        Повторная запись того же индекса перезаписывает файл: это делает
        конвейер идемпотентным при повторном прогоне одного батча.
        """
        target = self.path / f"train_{batch_idx:04d}.csv.gz"

        if target.exists() and target.stat().st_size > 0:
            # Уже сохранён — не дублируем, но фиксируем актуальный размер.
            return int(len(frame))

        missing = [column for column in self.columns if column not in frame.columns]
        if missing:
            raise KeyError(
                f"В батче {batch_idx} отсутствуют колонки для хранения: {missing}"
            )

        payload = frame[self.columns]
        payload.to_csv(target, index=False, compression="gzip")
        self._write_index()
        return int(len(payload))

    def clear(self) -> None:
        """Удалить все накопленные части (используется при переинициализации)."""
        for _, item in self._parts():
            item.unlink(missing_ok=True)
        self._write_index()

    # -- метаданные ------------------------------------------------------

    def stats(self, window: int | None = None) -> dict[str, Any]:
        """Сведения о содержимом хранилища для метаданных и отчёта."""
        parts = self._parts()
        if window is not None and window > 0:
            parts = parts[-window:]
        return {
            "enabled": True,
            "path": str(self.path),
            "parts": len(parts),
            "batch_indices": [index for index, _ in parts],
            "rows": int(sum(self._row_count(item) for _, item in parts)),
            "columns": self.columns,
        }

    @staticmethod
    def _row_count(path: Path) -> int:
        """Число строк без полного чтения файла."""
        cache = path.parent / _INDEX_NAME
        if cache.is_file():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                if path.name in data:
                    return int(data[path.name])
            except (json.JSONDecodeError, OSError, ValueError):
                pass
        return int(len(pd.read_csv(path)))

    def _write_index(self) -> None:
        """Кэш соответствия «файл → число строк» для быстрых отчётов."""
        index = {item.name: self._count_lines(item) for _, item in self._parts()}
        (self.path / _INDEX_NAME).write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _count_lines(path: Path) -> int:
        """Число строк данных в gzip-CSV (минус заголовок)."""
        with pd.read_csv(path, chunksize=50_000) as reader:
            return int(sum(len(chunk) for chunk in reader))
