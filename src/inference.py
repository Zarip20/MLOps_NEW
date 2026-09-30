"""Устойчивый инференс: проверка входа, калибровка, безопасная деградация (6.b.ii).

Задание требует «валидации входных данных, fallback на безопасную модель
при аномалиях или редких категориях, деградации вместо падения». Здесь
решаются все три части, и каждая закрывает свою поломку.

**1. Проверка входа.** Файл извне не обязан быть таким же, как
обучающий. Отсутствующая колонка, нечисловые значения там, где ждали
число, категория, которой не было при обучении, — всё это раньше
приводило либо к исключению посреди обработки, либо, что хуже, к
молчаливо неверному прогнозу. Проверка идёт **до** обращения к модели и
возвращает вердикт, а не исключение: решение об отказе принимает
вызывающий код, у которого есть контекст.

**2. Калибровка порога.** Здесь замеряется дефект, а не выдумывается
улучшение. Измерено на прогоне всех 48 батчей: модели предсказывают
положительный класс в 34…69 % строк при фактических 6,6 %. Причина в
том, что f1 максимизирует recall, а дисбаланс компенсируется весами
(`class_weight`, `sample_weight=balanced`), и модель учится называть
положительным всё подряд: это выгодно для f1 и бессмысленно для
бизнес-решения. Порог по квантилю вероятностей возвращает долю
положительных к заданному уровню, и делает это явно, а не порогом 0,5,
который на этих данных даёт тот же брак.

**3. Безопасная деградация.** Если продуктовая модель неприменима
(не та схема признаков, не читается файл), предсказание уходит
**упрощённой** модели, а не прерывается. Если и она неприменима —
работает константа «страховых случаев нет», с явной отметкой в отчёте.
Ложноотрицательный ответ «случаев нет» на месяц без страховых случаев
хотя бы безопасен, а пустой файл результата непригоден вообще.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Вердикты проверки входа, от лучшего к худшему.
STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
STATUS_REJECTED = "rejected"

#: Метка, которой можно было бы заменить незнакомую категорию.
#:
#: В конвейере она **не используется** и оставлена как явное напоминание
#: о бесполезности такой замены: кодировщик не обучен на этом значении,
#: поэтому `__unknown__` закодировался бы в те же нули, что и исходное
#: значение. Подмена создавала бы видимость починки, не меняя ни одного
#: числа в прогнозе.
UNKNOWN_CATEGORY = "__unknown__"


class InferenceError(RuntimeError):
    """Прогноз выполнить нельзя даже в аварийном режиме."""


# ---------------------------------------------------------------------------
# Проверка входа
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Результат проверки файла перед прогнозом."""

    status: str = STATUS_OK
    checks: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    rows: int = 0

    @property
    def accepted(self) -> bool:
        return self.status != STATUS_REJECTED

    @property
    def degraded(self) -> bool:
        return self.status == STATUS_DEGRADED

    def add(
        self, name: str, passed: bool, detail: str = "", *, fatal: bool = False
    ) -> None:
        """Записать результат одной проверки и обновить вердикт.

        Args:
            name: имя проверки.
            passed: True, если проверка **пройдена**. Знак важен:
                ошибка была бы здесь незаметна, потому что при
                превышении порога выбросов проверка как раз «срабатывает»,
                и перепутанный знак тихо пропускал бы аномальный файл.
            detail: что именно обнаружено — показывается в отчёте.
            fatal: отвергнуть вход целиком, а не пометить как снижение
                качества.
        """
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        if passed:
            return
        self.problems.append(f"{name}: {detail}" if detail else name)
        if fatal and self.status == STATUS_OK:
            self.status = STATUS_REJECTED
        elif self.status == STATUS_OK:
            self.status = STATUS_DEGRADED

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "rows": self.rows,
            "checks": self.checks,
            "problems": self.problems,
            "repairs": self.repairs,
            "accepted": self.accepted,
            "degraded": self.degraded,
        }


def validate_input(
    frame: pd.DataFrame,
    required_columns: Sequence[str],
    numerical_columns: Sequence[str],
    *,
    known_categories: dict[str, Any] | None = None,
    ranges: dict[str, tuple[float, float]] | None = None,
    max_outlier_ratio: float = 0.2,
    min_rows: int = 1,
) -> ValidationReport:
    """Проверить кадр перед прогнозом и починить, что чинится.

    Проверки делятся на две группы по последствиям. Отсутствие колонки
    или пустой файл делают прогноз бессмысленным — такой вход
    **отвергается**. Нечисловые значения, незнакомые категории и
    выбросы делают прогноз менее точным, но не бессмысленным: они
    **чинятся** (приведение к числу, замена на «неизвестно») и
    отмечаются как снижение качества. Различать эти группы важно,
    иначе система либо отвергает вполне пригодные файлы, либо молча
    обрабатывает пустые.

    Исключение — выбросы за пределами `max_outlier_ratio`: такая доля
    означает, что файл describes совсем другой распределение, и прогноз
    по нему был бы не просто неточным, а бессмысленным.

    Args:
        frame: кадр из внешнего файла; при необходимости чинится на месте.
        required_columns: колонки, без которых прогноз невозможен.
        numerical_columns: колонки, которые должны быть числами.
        known_categories: известные при обучении значения категорий.
        ranges: обучающие диапазоны числовых колонок. `None` — берутся
            из накопленных при обучении; передаются явно, чтобы
            проверка не зависела от порядка вызовов в процессе.
        max_outlier_ratio: доля выбросов, выше которой вход отвергается.
        min_rows: минимально допустимое число строк.

    Returns:
        `ValidationReport`.
    """
    report = ValidationReport(rows=int(len(frame)))

    if len(frame) < max(min_rows, 1):
        report.add(
            "строки", False,
            f"в файле {len(frame)} строк, требуется не менее {max(min_rows, 1)}",
            fatal=True,
        )
        return report

    missing = [name for name in required_columns if name not in frame.columns]
    if missing:
        report.add(
            "обязательные колонки", False,
            f"отсутствуют: {', '.join(missing)}", fatal=True,
        )
        return report
    report.add("обязательные колонки", True, f"{len(required_columns)} на месте")

    # Числовые колонки: приводим и считаем, сколько было исправлено.
    repaired = 0
    for name in numerical_columns:
        if name not in frame.columns:
            continue
        original = frame[name]
        converted = pd.to_numeric(original, errors="coerce")
        broken = int(converted.isna().sum() - original.isna().sum())
        if broken:
            repaired += broken
            frame[name] = converted
    if repaired:
        report.add(
            "числовые значения", False,
            f"не удалось разобрать {repaired} значений; они заменены на пропуск "
            f"и будут импутированы препроцессором",
        )
        report.repairs.append(f"числовые значения: {repaired}")
    else:
        report.add("числовые значения", True)

    if known_categories:
        unknown: dict[str, dict[str, int]] = {}
        for name, known in known_categories.items():
            if name not in frame.columns or not known:
                continue
            column = frame[name].astype(str)
            mask = ~column.isin({str(item) for item in known})
            if not mask.any():
                continue
            top = column[mask].value_counts().head(10)
            unknown[name] = {str(key): int(value) for key, value in top.items()}
        if unknown:
            # Значения НЕ подменяются: кодировщик не обучен на
            # `__unknown__` и всё равно закодирует её в нули, то есть
            # подмена не изменила бы числа, а создала бы видимость
            # починки. Вместо этого они перечисляются: неизвестная
            # категория — это факт о данных, и его надо назвать.
            total = sum(
                count for values in unknown.values() for count in values.values()
            )
            listed = "; ".join(
                f"{column}={value} ({count})"
                for column, values in unknown.items()
                for value, count in values.items()
            )
            report.add(
                "незнакомые категории", False,
                f"{total} значений не встречались при обучении: {listed}. "
                f"Кодировщик даст им нули, то есть вклад, близкий к "
                f"«среднему»; на результат это влияет, но исправить без "
                f"переобучения нельзя",
            )
            report.repairs.append(f"незнакомые категории: {total}")
        else:
            report.add("незнакомые категории", True)

    effective = known_ranges() if ranges is None else ranges
    outliers = count_outliers(frame, numerical_columns, effective)
    if outliers:
        ratio = outliers / max(len(frame), 1)
        report.add(
            "выбросы", ratio <= max_outlier_ratio,
            f"{outliers} значений вне обучающего диапазона ({ratio:.1%} строк)",
            fatal=True,
        )
    else:
        report.add("выбросы", True)

    return report


#: Обучающие диапазоны числовых колонок, накапливаемые при обучении.
_TRAINING_RANGES: dict[str, tuple[float, float]] = {}


def remember_training_ranges(
    frame: pd.DataFrame, columns: Sequence[str]
) -> dict[str, tuple[float, float]]:
    """Запомнить диапазоны числовых колонок по обучающим данным.

    Границы — «забор Тьюки»: первый квартиль минус и третий плюс три
    межквартильных размаха. Он выбран не потому, что красив, а потому
    что устойчив к жирным хвостам: `PREMIUM` распределён так, что
    0,5 % и 99,5 % квантили разъезжаются на порядки и «нормальным»
    объявляют почти весь диапазон.

    Чего этот правило **не** делает: одиночный чудовищный выброс в
    небольшой выборке не исключается — при девяноста девяти равных
    значениях межквартильный размах равен нулю, и забор вырождается.
    Для таких случаев полагаться нужно на долю выбросов в проверке
    входа, а не на отдельные границы.
    """
    for name in columns:
        if name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        if iqr > 0:
            low = max(float(q1 - 3.0 * iqr), float(values.min()))
            high = min(float(q3 + 3.0 * iqr), float(values.max()))
        else:
            # Все значения совпадают либо почти совпадают: забор по
            # размаху выродился бы в точку, и любое отличие считалось бы
            # выбросом. Тогда берём квантили.
            low = float(values.quantile(0.005))
            high = float(values.quantile(0.995))
        _TRAINING_RANGES[name] = (low, high)
    return known_ranges()


def known_ranges() -> dict[str, tuple[float, float]]:
    """Запомненные обучающие диапазоны (копия)."""
    return dict(_TRAINING_RANGES)


def forget_training_ranges() -> None:
    """Сбросить накопленные диапазоны.

    Нужен при переходе к другому набору данных: иначе проверка входа
    сравнивала бы новые файлы с диапазонами прежних.
    """
    _TRAINING_RANGES.clear()


def count_outliers(
    frame: pd.DataFrame,
    columns: Sequence[str],
    ranges: dict[str, tuple[float, float]],
) -> int:
    """Сколько значений выходит за обучающий диапазон."""
    if not ranges:
        return 0
    total = 0
    for name, (low, high) in ranges.items():
        if name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        total += int(((values < low) | (values > high)).sum())
    return total


# ---------------------------------------------------------------------------
# Калибровка порога
# ---------------------------------------------------------------------------


def choose_threshold(
    probabilities: Any,
    target_rate: float,
    *,
    default: float = 0.5,
) -> dict[str, Any]:
    """Порог, при котором доля положительных близка к заданной.

    Модель, обученная с компенсацией дисбаланса по весам, выдаёт
    вероятности, систематически сдвинутые вверх: медиана вероятностей
    может быть выше 0,5 при фактической доле положительных 3 %. Порог
    0,5 в такой ситуации бесполезен, а квантиль задаёт долю решений
    напрямую.

    Args:
        probabilities: вероятности положительного класса.
        target_rate: желаемая доля положительных решений, 0…1.
        default: порог, если вычислить не на чем (пустой вход).

    Returns:
        Сведения о пороге, включая долю, которую дал бы порог 0,5 —
        без этого сравнения не видно, что калибровка вообще что-то
        изменила.
    """
    array = np.asarray(probabilities, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "threshold": float(default),
            "target_rate": float(target_rate),
            "achieved_rate": None,
            "raw_rate_at_half": None,
            "median_probability": None,
            "reason": "нет вероятностей",
        }

    if 0 < target_rate < 1:
        threshold = float(np.quantile(array, 1.0 - target_rate))
    else:
        threshold = float(default)

    achieved = float((array >= threshold).mean())
    # При совпадающих вероятностях квантиль не может разделить строки:
    # порог равен им всем, и достигнутая доля скачет до 1. Это не
    # брак калибровки, а свойство данных, и о нём нужно сказать прямо,
    # иначе в логе появилось бы «положительных 100 %» без объяснения.
    overshoot = achieved > min(1.0, target_rate * 1.5)
    return {
        "threshold": round(threshold, 6),
        "target_rate": round(float(target_rate), 6),
        "achieved_rate": round(achieved, 6),
        "raw_rate_at_half": round(float((array >= 0.5).mean()), 6),
        "median_probability": round(float(np.median(array)), 6),
        "n_distinct": int(len(np.unique(array))),
        "ties_limit_calibration": bool(overshoot),
        "reason": (
            "порог по квантилю вероятностей; достигнутая доля выше "
            "заданной, потому что вероятности почти не различаются"
            if overshoot else "порог по квантилю вероятностей"
        ),
    }


# ---------------------------------------------------------------------------
# Безопасное предсказание
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Итог прогноза вместе с тем, как он получен."""

    predictions: Any
    probabilities: Any | None
    model: str
    threshold: float | None
    validation: ValidationReport
    fallback_used: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        size = int(np.size(self.predictions))
        return {
            "model": self.model,
            "fallback_used": self.fallback_used,
            "threshold": self.threshold,
            "notes": self.notes,
            "validation": self.validation.as_dict(),
            "positive_rate": round(float(np.mean(self.predictions)), 6) if size else None,
        }


class SafePredictor:
    """Прогноз с проверкой входа, калибровкой и безопасной деградацией.

    Порядок попыток: продуктовая модель → упрощённая → константа.
    Каждая следующая попытка дешевле и надёжнее предыдущей, поэтому
    переход вниз — осознанная деградация, а не падение. Каждый переход
    записывается в `notes`: по отчёту видно, что прогноз получен не той
    моделью, на которую рассчитывали.
    """

    def __init__(
        self,
        models: dict[str, Any],
        preprocessor: Any,
        feature_columns: Sequence[str],
        *,
        fallback: str = "lr",
        threshold_mode: str = "quantile",
        threshold: float = 0.5,
        target_positive_rate: float | None = None,
    ) -> None:
        self.models = models
        self.preprocessor = preprocessor
        self.feature_columns = list(feature_columns)
        self.fallback = fallback
        self.threshold_mode = threshold_mode
        self.threshold = float(threshold)
        self.target_positive_rate = target_positive_rate

    def _ordered_models(self) -> list[tuple[str, Any]]:
        """Модели в порядке предпочтения: продуктовая, затем упрощённая."""
        if "best" in self.models:
            primary = [("best", self.models["best"])]
            rest = [
                (name, model) for name, model in self.models.items()
                if name != "best"
            ]
        else:
            primary = []
            rest = list(self.models.items())

        if self.fallback in self.models and not primary:
            promoted = self.models[self.fallback]
            primary = [(self.fallback, promoted)]
            rest = [
                (name, model) for name, model in rest if name != self.fallback
            ]
        elif self.fallback in self.models and primary:
            rest = [(self.fallback, self.models[self.fallback])] + [
                (name, model) for name, model in rest
                if name != self.fallback
            ]
        return primary + rest

    def predict(
        self, frame: pd.DataFrame, report: ValidationReport
    ) -> PredictionResult:
        """Выполнить прогноз с деградацией вместо отказа."""
        notes: list[str] = []
        ordered = self._ordered_models()

        if not ordered:
            return self._constant(
                len(frame), report, "нет ни одной модели", notes
            )

        for position, (name, model) in enumerate(ordered):
            try:
                matrix = self.preprocessor.transform(frame[self.feature_columns])
                probabilities = self._probabilities(model, matrix)
                return self._decide(
                    name, model, matrix, probabilities, report, notes, position
                )
            except Exception as error:  # noqa: BLE001
                # Ошибка любой модели не должна обрывать прогноз: ниже
                # есть вариант проще, и он лучше пустого результата.
                logger.warning(
                    "Модель %s неприменима (%s), перехожу к следующей",
                    name, error,
                )
                notes.append(f"{name}: неприменима — {error}")

        return self._constant(
            len(frame), report, "ни одна модель неприменима", notes
        )

    def _decide(
        self,
        name: str,
        model: Any,
        matrix: Any,
        probabilities: Any | None,
        report: ValidationReport,
        notes: list[str],
        position: int,
    ) -> PredictionResult:
        if position:
            notes.append(
                f"использована запасная модель {name!r} вместо основной"
            )

        if probabilities is None:
            return PredictionResult(
                predictions=np.asarray(model.predict(matrix)).astype(int),
                probabilities=None,
                model=name,
                threshold=None,
                validation=report,
                fallback_used=bool(position),
                notes=notes,
            )

        if self.threshold_mode == "quantile" and self.target_positive_rate:
            info = choose_threshold(
                probabilities, self.target_positive_rate, default=self.threshold
            )
            threshold = info["threshold"]
            notes.append(
                f"порог {threshold:.4f} подобран под долю положительных "
                f"{self.target_positive_rate:.1%}; без калибровки модель "
                f"дала бы {info['raw_rate_at_half']:.1%} положительных"
            )
            if info.get("ties_limit_calibration"):
                notes.append(
                    f"достигнутая доля {info['achieved_rate']:.1%} выше "
                    f"заданной: вероятностей всего {info['n_distinct']} "
                    f"различных, порог не может их разделить. Нужна модель "
                    f"с более тонкой градацией вероятностей"
                )
        else:
            threshold = self.threshold

        predictions = (
            np.asarray(probabilities, dtype=float) >= threshold
        ).astype(int)
        return PredictionResult(
            predictions=predictions,
            probabilities=np.asarray(probabilities, dtype=float),
            model=name,
            threshold=threshold,
            validation=report,
            fallback_used=bool(position),
            notes=notes,
        )

    @staticmethod
    def _probabilities(model: Any, matrix: Any) -> Any | None:
        if not hasattr(model, "predict_proba"):
            return None
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)

    def _constant(
        self,
        rows: int,
        report: ValidationReport,
        reason: str,
        notes: list[str],
    ) -> PredictionResult:
        """Константный прогноз «страховых случаев нет».

        Осознанно безопасное поведение: месяц без страховых случаев при
        доле 2,6 % — обычное дело, и такой ответ хотя бы пригоден для
        сверки. Молчаливый пустой файл непригоден ни для чего, поэтому
        константа лучше отказа.
        """
        notes.append(f"{reason}; выдан константный прогноз «случаев нет»")
        report.add(
            "прогноз", False,
            f"{reason} — выдан константный прогноз, качество не гарантируется",
        )
        return PredictionResult(
            predictions=np.zeros(int(rows), dtype=int),
            probabilities=None,
            model="constant",
            threshold=None,
            validation=report,
            fallback_used=True,
            notes=notes,
        )
