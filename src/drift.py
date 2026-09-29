"""Обнаружение дрейфа данных и модели (2.b.iv, 5.b.ii).

Зачем это нужно именно здесь. В наборе доля страховых случаев падает
с 11.0 % (2014-07) до 2.6 % (2018-06) — это дрейф целевой метки, и он
объясняет падение f1 с 0.475 до 0.221 в baseline-прогоне. Без контроля
дрейфа падение качества выглядит как «модель испортилась», и лечится
переобучением, которое не помогает: данные изменились, а модель —
нет.

Реализованы два механизма:

* **PSI** (Population Stability Index) для числовых признаков,
  распределения целевой метки и категориальных признаков. Показывает,
  насколько распределение сместилось относительно обучающего окна.
* **KS-статистика** для числовых признаков: чувствительнее PSI к
  небольшим сдвигам и даёт понятную величину эффекта.

Оба метода считаются относительно **обучающего окна**, а не относительно
предыдущего батча: только так можно отличить плановый сдвиг от
случайного. Результат не прерывает конвейер — при превышении порога
ставится статус и пишется в метаданные (АР-10).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Число интервалов при построении гистограммы для PSI.
DEFAULT_BINS = 10

#: Уровни интерпретации PSI по общепринятой шкале.
PSI_LEVELS = (
    (0.10, "stable", "стабильно"),
    (0.25, "moderate", "умеренный сдвиг"),
    (float("inf"), "significant", "значительный сдвиг"),
)


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = DEFAULT_BINS,
    epsilon: float = 1e-6,
) -> float:
    """PSI между двумя распределениями.

    Both distributions are split into equal-width bins computed on the
    *expected* sample, so the two histograms are comparable. Empty bins
    are replaced by a small constant: a zero share would otherwise make
    the logarithm blow up exactly when the shift is largest.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if expected.size == 0 or actual.size == 0:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if edges.size < 3:
        # Слишком мало различных значений для разбиения на интервалы.
        return float("nan")

    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)

    expected_share = expected_counts / max(expected.size, 1)
    actual_share = actual_counts / max(actual.size, 1)

    expected_share = np.clip(expected_share, epsilon, None)
    actual_share = np.clip(actual_share, epsilon, None)

    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def kolmogorov_smirnov(expected: np.ndarray, actual: np.ndarray) -> float:
    """Двусторонняя KS-статистика между двумя выборками."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if expected.size == 0 or actual.size == 0:
        return float("nan")

    combined = np.sort(np.concatenate([expected, actual]))
    cdf_expected = np.searchsorted(np.sort(expected), combined, side="right") / expected.size
    cdf_actual = np.searchsorted(np.sort(actual), combined, side="right") / actual.size
    return float(np.max(np.abs(cdf_expected - cdf_actual)))


def interpret_psi(psi: float) -> tuple[str, str]:
    """Перевести числовой PSI в уровень и пояснение."""
    if not np.isfinite(psi):
        return "unknown", "недостаточно данных"
    for threshold, level, description in PSI_LEVELS:
        if psi < threshold:
            return level, description
    return "significant", "значительный сдвиг"


class DriftMonitor:
    """Сравнивает текущий батч с обучающим окном.

    Обучающее окно накапливается по мере обработки батчей и хранится
    отдельно от накопительного хранилища обучения: для дрейфа важен
    состав признаков, а не состав обучающей выборки, и смешивать эти
    две вещи не следует.
    """

    def __init__(
        self,
        thresholds: dict[str, Any] | None = None,
        bins: int = DEFAULT_BINS,
        max_categorical: int = 12,
    ) -> None:
        self.thresholds = thresholds or {}
        self.bins = bins
        self.max_categorical = max_categorical
        # PSI по категориальным признакам выключен: на проверенных данных
        # он давал неверные значения (0.98 для колонки SEX с тремя
        # категориями и неизменными долями). Категории отслеживаются по
        # появлению новых значений — эта мера корректна.
        self.categorical_psi_enabled = bool(
            self.thresholds.get("categorical_psi", False)
        )
        self._reference: dict[str, Any] = {}

    # -- формирование эталона -------------------------------------------

    def set_reference(
        self,
        numerical_cols: Sequence[str],
        categorical_cols: Sequence[str],
        target: str,
        frame: pd.DataFrame,
    ) -> None:
        """Задать эталон из первого батча (или заданного окна)."""
        self._reference = {
            "numerical": {
                column: self._numeric_reference(frame[column])
                for column in numerical_cols
                if column in frame.columns
            },
            "categorical": {
                column: self._categorical_reference(frame[column])
                for column in categorical_cols
                if column in frame.columns
            },
            "target": self._numeric_reference(frame[target]) if target in frame.columns else None,
            "rows": int(len(frame)),
        }
        logger.info(
            "Эталон дрейфа задан: %d числовых, %d категориальных признаков",
            len(self._reference["numerical"]),
            len(self._reference["categorical"]),
        )

    def has_reference(self) -> bool:
        return bool(self._reference)

    @staticmethod
    def _numeric_reference(series: pd.Series) -> dict[str, Any]:
        values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            return {"values": np.array([]), "n": 0}
        if values.size > 20000:
            # Равномерная выборка: эталон должен помещаться в память
            # и не зависеть от размера батча.
            step = values.size // 20000 + 1
            values = values[::step]
        return {"values": values, "n": int(values.size)}

    @staticmethod
    def _categorical_reference(series: pd.Series) -> dict[str, Any]:
        shares = series.dropna().astype(str).value_counts(normalize=True)
        return {"shares": shares.to_dict(), "n": int(series.notna().sum())}

    def extend_reference(
        self,
        frame: pd.DataFrame,
        numerical_cols: Sequence[str],
        categorical_cols: Sequence[str],
        target: str,
        window_batches: int = 3,
    ) -> None:
        """Сдвинуть эталон на текущий батч, сохранив скользящее окно.

        Эталон не должен быть «заморожен» на первом батче: иначе уже
        естественный сезонный сдвиг 2014 → 2018 года будет выглядеть
        как постоянная аномалия. Но и накапливать его нельзя — тогда
        эталон размывается и перестаёт соответствовать текущему периоду.

        Хранится именно **окно** из последних `window_batches` батчей, а
        не вся история. Это принципиально для месячных данных с выраженной
        сезонностью: при сравнении соседних месяцев PSI и так велик, и
        добавление истории в эталон увеличивало бы его ещё сильнее
        (наблюдалось 0.92 в среднем, 1.76 в максимуме).
        """
        if not self._reference:
            return

        limit = max(int(window_batches), 1) * 20000

        for column, reference in self._reference["numerical"].items():
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(
                    dtype=float
                )
                if values.size:
                    # Равномерно прореживаем, чтобы размер эталона не зависел
                    # от размера батча.
                    if values.size > 20000:
                        values = values[:: max(values.size // 20000, 1)]
                    reference["values"] = np.concatenate(
                        [reference["values"], values]
                    )[-limit:]

        for column, reference in self._reference["categorical"].items():
            if column in frame.columns:
                current = frame[column].dropna().astype(str).value_counts(normalize=True)
                merged = pd.concat([pd.Series(reference["shares"]), current]).groupby(
                    level=0
                ).mean()
                reference["shares"] = merged.to_dict()

    # -- проверка --------------------------------------------------------

    def check(
        self,
        frame: pd.DataFrame,
        target: str,
        numerical_cols: Sequence[str],
        categorical_cols: Sequence[str],
    ) -> dict[str, Any]:
        """Оценить дрейф текущего батча относительно эталона."""
        if not self._reference:
            return {
                "available": False,
                "reason": "эталон не задан",
                "status": "unknown",
            }

        thresholds = self.thresholds
        warn = float(thresholds.get("psi_warn", 0.10))
        alert = float(thresholds.get("psi_alert", 0.25))
        ks_warn = float(thresholds.get("ks_warn", 0.20))

        numerical: dict[str, Any] = {}
        worst_psi = 0.0
        worst_ks = 0.0

        for column, reference in self._reference["numerical"].items():
            if column not in frame.columns:
                continue
            actual = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(
                dtype=float
            )
            psi = population_stability_index(reference["values"], actual, self.bins)
            ks = kolmogorov_smirnov(reference["values"], actual)
            level, description = interpret_psi(psi)
            numerical[column] = {
                "psi": round(psi, 4) if np.isfinite(psi) else None,
                "ks": round(ks, 4) if np.isfinite(ks) else None,
                "level": level,
                "reference_median": round(float(np.median(reference["values"])), 4)
                if reference["values"].size else None,
                "actual_median": round(float(np.median(actual)), 4) if actual.size else None,
            }
            if np.isfinite(psi):
                worst_psi = max(worst_psi, float(psi))
            if np.isfinite(ks):
                worst_ks = max(worst_ks, float(ks))

        categorical: dict[str, Any] = {}
        # PSI по категориям отключён по умолчанию. На проверенных данных
        # он давал 0.98 для колонки SEX с тремя значениями (0/1/2) и
        # практически неизменными долями, то есть метрика была неверной.
        # Причина — усреднение долей скользящего эталона: доли редких
        # категорий «размываются», и сумма разностей перестаёт отражать
        # реальный сдвиг. Показывать заведомо неправильное число хуже,
        # чем не показывать ничего.
        #
        # Вместо него по категориям отслеживается появление новых
        # значений: на поздних батчах действительно встречаются марки,
        # не попавшие в эталон.
        if self.categorical_psi_enabled:
            for column, reference in list(self._reference["categorical"].items())[
                : self.max_categorical
            ]:
                if column not in frame.columns:
                    continue
                current = frame[column].dropna().astype(str).value_counts(normalize=True)
                psi = _categorical_psi(reference["shares"], current.to_dict())
                level, _ = interpret_psi(psi)
                categorical[column] = {
                    "psi": round(psi, 4) if np.isfinite(psi) else None,
                    "level": level,
                }

        new_category_warn = float(thresholds.get("new_category_share_warn", 0.01))
        for column, reference in self._reference["categorical"].items():
            if column not in frame.columns:
                continue
            current = frame[column].dropna().astype(str)
            new_values = sorted(set(current.unique()) - set(reference["shares"]))
            if not new_values:
                continue
            share = float(current.isin(new_values).sum() / max(len(current), 1))
            entry = categorical.setdefault(column, {})
            entry["new_categories"] = new_values[:8]
            entry["new_categories_share"] = round(share, 5)
            if share > new_category_warn:
                entry["level"] = "warn"
                entry["note"] = f"{share:.1%} значений отсутствовали в эталоне"

        target_drift: dict[str, Any] = {}
        reference_target = self._reference.get("target")
        if reference_target and target in frame.columns and reference_target["values"].size:
            actual = pd.to_numeric(frame[target], errors="coerce").dropna().to_numpy(
                dtype=float
            )
            reference_rate = float((reference_target["values"] == 1).mean())
            actual_rate = float((actual == 1).mean()) if actual.size else 0.0

            # Для метки используется бинарный PSI: квантильная гистограмма
            # по переменной из двух значений вырождается в NaN.
            psi = binary_psi(reference_rate, actual_rate)
            level, description = interpret_psi(psi)
            target_drift = {
                "psi": round(psi, 4) if np.isfinite(psi) else None,
                "level": level,
                "description": description,
                "reference_positive_rate": round(reference_rate, 6),
                "actual_positive_rate": round(actual_rate, 6),
                "relative_change": (
                    round((actual_rate / reference_rate - 1) * 100, 1)
                    if reference_rate
                    else None
                ),
            }
            if np.isfinite(psi):
                worst_psi = max(worst_psi, float(psi))

        status = "ok"
        if worst_psi >= alert:
            status = "alert"
        elif worst_psi >= warn or worst_ks >= ks_warn:
            status = "warn"

        return {
            "available": True,
            "status": status,
            "max_psi": round(worst_psi, 4),
            "max_ks": round(worst_ks, 4),
            "thresholds": {
                "psi_warn": warn,
                "psi_alert": alert,
                "ks_warn": ks_warn,
                "new_category_share_warn": new_category_warn,
                "categorical_psi": self.categorical_psi_enabled,
            },
            "numerical": numerical,
            "categorical": categorical,
            "target": target_drift,
            "note": (
                "Дрейф не прерывает конвейер: результат фиксируется в "
                "метаданных и влияет на решение гейта качества (АР-10)."
            ),
        }


def _categorical_psi(
    reference: dict[str, float],
    actual: dict[str, float],
    rare_ratio: float = 0.01,
    epsilon: float = 1e-4,
) -> float:
    """PSI для категориального распределения.

    Редкие категории схлопываются в одну группу с последующей нормировкой
    долей. Без этого PSI на колонке вроде `MAKE` (747 значений)
    нечитаем: каждая из сотен редких категорий сменяет долю на величину
    порядка `epsilon`, и сумма разностей даёт PSI больше единицы независимо
    от того, менялось ли распределение. Именно это давало `max_psi` = 0.93
    при максимуме 0.39 по числовым признакам.
    """
    if not reference and not actual:
        return float("nan")

    total_reference = sum(reference.values()) or 1.0
    total_actual = sum(actual.values()) or 1.0

    common = set(reference) | set(actual)
    rare = {
        key
        for key in common
        if reference.get(key, 0.0) / total_reference < rare_ratio
        and actual.get(key, 0.0) / total_actual < rare_ratio
    }

    groups = sorted(key for key in common if key not in rare)
    if not groups:
        return float("nan")

    expected = np.array(
        [sum(reference.get(key, 0.0) for key in rare) / total_reference]
        + [reference.get(key, 0.0) / total_reference for key in groups]
    )
    observed = np.array(
        [sum(actual.get(key, 0.0) for key in rare) / total_actual]
        + [actual.get(key, 0.0) / total_actual for key in groups]
    )

    expected = np.clip(expected, epsilon, None)
    observed = np.clip(observed, epsilon, None)
    expected = expected / expected.sum()
    observed = observed / observed.sum()

    return float(np.sum((observed - expected) * np.log(observed / expected)))


def binary_psi(
    reference_positive_rate: float,
    actual_positive_rate: float,
    epsilon: float = 1e-6,
) -> float:
    """PSI для бинарной метки.

    Обычная гистограмма с квантильными интервалами для переменной,
    принимающей два значения, вырождается: интервалов получается меньше
    трёх, и функция возвращает `nan`. Распределение метки бинарное, и PSI
    считается по двум долям — «положительный» и «отрицательный».
    """
    # Ожидаемое и фактическое распределения строятся РАЗДЕЛЬНО: обе
    # доли берутся из своих наборов. Сборка обоих распределений из одного
    # массива давала нулевую разность при любом сдвиге.
    expected = np.clip(
        np.array([reference_positive_rate, 1.0 - reference_positive_rate]),
        epsilon,
        1.0,
    )
    actual = np.clip(
        np.array([actual_positive_rate, 1.0 - actual_positive_rate]),
        epsilon,
        1.0,
    )
    expected = expected / expected.sum()
    actual = actual / actual.sum()

    return float(np.sum((actual - expected) * np.log(actual / expected)))
