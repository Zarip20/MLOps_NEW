"""Вспомогательные функции: сериализация, измерение времени, окружение.

Модуль не читает конфигурацию и не обращается к диску за пределами
явно переданных путей (АР-13) — это позволяет использовать его в тестах
и в отчётах без побочных эффектов.
"""

from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Сериализация
# ---------------------------------------------------------------------------


def to_json_serializable(value: Any) -> Any:
    """Привести значение к типу, допустимому в JSON.

    Обрабатываются numpy-скаляры (в том числе булевы), временные метки,
    пути и периоды pandas. Отдельная функция нужна потому, что
    `json.dump` по умолчанию падает на `numpy.bool_` — а именно такие
    значения возвращает sklearn.
    """
    if isinstance(value, dict):
        return {str(key): to_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_serializable(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        # JSON не допускает NaN и бесконечностей.
        return result if np.isfinite(result) else None
    if isinstance(value, np.ndarray):
        return [to_json_serializable(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, (date,)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NaT or value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path: str | os.PathLike[str], data: Any, indent: int = 2) -> Path:
    """Атомарно записать JSON с преобразованием типов."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    temporary = target.with_suffix(target.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(to_json_serializable(data), handle, ensure_ascii=False, indent=indent)
    os.replace(temporary, target)
    return target


def load_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    """Прочитать JSON; отсутствующий или повреждённый файл даёт `default`."""
    source = Path(path)
    if not source.is_file():
        return default
    try:
        with open(source, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return default


def read_artifacts(
    directory: str | os.PathLike[str], prefix: str
) -> list[dict[str, Any]]:
    """Прочитать все JSON-артефакты `prefix_NNNN.json` по возрастанию NNNN.

    Метаданные батчей пишутся по одному файлу на батч, и читать их надо
    в хронологическом порядке. Номер батча берётся из последнего
    компонента имени, а время изменения файла для этого не годится: в
    CI пересборка меняет его у всех файлов разом.

    Повреждённый или нечитаемый файл пропускается, а не обрывает чтение:
    один битый артефакт иначе лишил бы отчёта всех остальных батчей.
    """
    folder = Path(directory)
    if not folder.is_dir():
        return []

    items: list[tuple[int, dict[str, Any]]] = []
    for path in folder.glob(f"{prefix}_*.json"):
        try:
            index = int(path.stem.rsplit("_", 1)[-1])
        except (IndexError, ValueError):
            continue
        payload = load_json(path)
        if isinstance(payload, dict):
            items.append((index, payload))
    return [payload for _, payload in sorted(items, key=lambda pair: pair[0])]


# ---------------------------------------------------------------------------
# Сериализация моделей
# ---------------------------------------------------------------------------


def atomic_pickle_dump(obj: Any, path: str | os.PathLike[str]) -> Path:
    """Сериализовать объект атомарно.

    Запись идёт во временный файл с последующим переименованием. Прямая
    запись опасна: если процесс прервётся после усечения файла, но до
    конца `pickle.dump`, на диске останется пустой или обрезанный артефакт,
    который при следующем запуске приведёт к `EOFError`. Такое уже
    случалось при отладке конвейера.
    """
    import pickle

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    temporary = target.with_name(target.name + ".tmp")
    try:
        with open(temporary, "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def safe_pickle_load(path: str | os.PathLike[str]) -> Any:
    """Загрузить объект; при повреждённом файле поднять понятную ошибку.

    Args:
        path: путь к файлу.

    Raises:
        FileNotFoundError: файла нет.
        ValueError: файл существует, но не читается как pickle.
    """
    import pickle

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Файл не найден: {source}")

    try:
        with open(source, "rb") as handle:
            return pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as error:
        raise ValueError(
            f"Не удалось загрузить {source.name}: файл повреждён ({error}). "
            f"Удалите его — артефакт будет создан заново на следующем прогоне."
        ) from error


def prune_none(payload: dict[str, Any]) -> dict[str, Any]:
    """Убрать ключи со значением `None` — отчёт становится читаемее."""
    return {key: value for key, value in payload.items() if value is not None}


# ---------------------------------------------------------------------------
# Измерение времени и памяти
# ---------------------------------------------------------------------------


class Stopwatch:
    """Накопительный измеритель времени по именованным этапам."""

    def __init__(self) -> None:
        self.durations: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.durations[name] = round(
                self.durations.get(name, 0.0) + time.perf_counter() - started, 3
            )

    def snapshot(self) -> dict[str, float]:
        """Снимок накопленных значений.

        Нужен, чтобы записать в манифест батча время именно этого батча:
        счётчик живёт дольше одного батча, и без снимка в каждом
        манифесте оказывалась бы накопленная сумма с начала прогона.
        """
        return dict(self.durations)

    def delta(self, snapshot: dict[str, float]) -> dict[str, float]:
        """Приращение относительно снимка."""
        return {
            key: round(value - snapshot.get(key, 0.0), 3)
            for key, value in self.durations.items()
        }

    def as_dict(self) -> dict[str, float]:
        return dict(self.durations)


def _windows_peak_memory_mb() -> float | None:
    """Пиковое потребление памяти процесса на Windows, МБ.

    Требует явных `argtypes`: без них ctypes передаёт дескриптор
    процесса как 32-битное целое, и `GetProcessMemoryInfo` завершается
    ошибкой, возвращая нули (что выглядит как «память не измеряется»).
    """
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE

    for library in ("psapi", "kernel32"):
        try:
            module = ctypes.WinDLL(library, use_last_error=True)
        except OSError:
            continue
        for symbol in ("GetProcessMemoryInfo", "K32GetProcessMemoryInfo"):
            try:
                function = getattr(module, symbol)
            except AttributeError:
                continue
            function.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            function.restype = wintypes.BOOL

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(ProcessMemoryCounters)
            if function(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return round(counters.PeakWorkingSetSize / (1024 * 1024), 1)
    return None


def peak_memory_mb() -> dict[str, float | None]:
    """Потребление памяти процесса.

    Returns:
        Словарь с двумя источниками, где каждый может быть `None`:
        `peak_rss_mb` — пиковое потребление памяти ОС (реальная память
        процесса, доступно не на всех платформах);
        `peak_python_mb` — пик кучи Python, который измеряется везде,
        но не учитывает память numpy-буферов и интерпретатора.
        Пустое значение означает, что источник недоступен на этой
        платформе, а не что память не измерялась.
    """
    rss: float | None = None
    try:
        import resource  # POSIX
    except ImportError:
        rss = _windows_peak_memory_mb()
    else:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux отдаёт килобайты, macOS — байты.
        divisor = 1024 if sys.platform != "darwin" else 1024 * 1024
        rss = round(usage / divisor, 1)

    python_heap: float | None = None
    try:
        import tracemalloc

        if tracemalloc.is_tracing():
            python_heap = round(tracemalloc.get_traced_memory()[1] / (1024 * 1024), 1)
    except ImportError:
        pass

    return {"peak_rss_mb": rss, "peak_python_mb": python_heap}


# ---------------------------------------------------------------------------
# Окружение
# ---------------------------------------------------------------------------


def describe_environment() -> dict[str, Any]:
    """Сведения об окружении для манифеста прогона (АР-6)."""
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def set_seed(seed: int) -> None:
    """Зафиксировать генераторы случайных чисел для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import sklearn

        sklearn.set_config(assume_finite=False)
    except ImportError:
        pass


def format_duration(seconds: float) -> str:
    """Человекочитаемая длительность."""
    if seconds < 1:
        return f"{seconds * 1000:.0f} мс"
    if seconds < 60:
        return f"{seconds:.1f} с"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes} мин {rest} с"


# ---------------------------------------------------------------------------
# Даты
# ---------------------------------------------------------------------------


def parse_datetime(
    series: pd.Series,
    date_format: str | None = None,
    min_parsed_ratio: float = 0.5,
) -> pd.Series:
    """Разобрать даты, переходя к другим форматам при неудаче.

    Формат в конфигурации описывает **исходный** набор (`%d-%b-%y`), а
    батчи после разбиения записываются в ISO. Жёсткое применение одного
    формата молча ломало всё, что зависит от дат: метрика своевременности
    возвращала «не удалось разобрать даты», а календарные производные
    признаки получались полностью пустыми и выбрасывались импутером.

    Порядок попыток: заданный формат → ISO 8601 → определение по данным.
    Переход происходит, только если предыдущий способ разобрал меньше
    половины значений, то есть когда он заведомо не подходит.
    """
    if series.empty:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    if date_format:
        candidate = pd.to_datetime(series, errors="coerce", format=date_format)
        if candidate.notna().sum() >= len(series) * min_parsed_ratio:
            return candidate
        parsed = candidate

    iso = pd.to_datetime(series, errors="coerce", format="ISO8601")
    if iso.notna().sum() >= len(series) * min_parsed_ratio:
        return iso
    if iso.notna().sum() > parsed.notna().sum():
        parsed = iso

    inferred = pd.to_datetime(series, errors="coerce", format="mixed")
    if inferred.notna().sum() > parsed.notna().sum():
        return inferred
    return parsed
