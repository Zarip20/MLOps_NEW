"""Загрузка, валидация и разрешение конфигурации (АР-1).

Конфигурация — единственный источник истины: ни пути, ни пороги, ни
гиперпараметры не должны быть зашиты в коде. Модуль отвечает за:

* чтение и валидацию `config.yaml` по схеме с понятными сообщениями об ошибках;
* разрешение относительных путей относительно корня проекта;
* вычисление хэша конфигурации (`config_hash`) для прослеживаемости артефактов (АР-6).

Конфигурация загружается один раз при старте и передаётся дальше явно
(АР-13: никаких `load_config()` в горячих местах и на уровне модулей).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils import load_json

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_NAME = "config.yaml"

#: Ключи верхнего уровня, обязательные для работы конвейера.
REQUIRED_SECTIONS = (
    "paths",
    "batching",
    "features",
    "target",
    "quality",
    "preprocessing",
    "models",
    "validation",
)

VALID_OPERATORS = (
    "eq", "ne", "ge", "gt", "le", "lt",
    "between", "not_between", "in", "not_in",
    "notna", "isna", "positive", "non_negative",
)

NUMERIC_OPERATORS = ("ge", "gt", "le", "lt", "between", "not_between", "positive", "non_negative")
SEVERITIES = ("error", "warn")


class ConfigError(ValueError):
    """Некорректный или неполный конфигурационный файл."""


@dataclass(frozen=True)
class Config:
    """Неизменяемая обёртка над словарём конфигурации.

    Хранит проверенные данные, корень проекта и хэш конфигурации. Все
    обращения идут через словарь (`cfg["quality"]`), а готовые подразделы
    доступны как свойства, чтобы не размазывать строковые ключи по коду.
    """

    data: dict[str, Any]
    root: Path

    # -- доступ к данным -------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    # -- разделы, которые нужны почти всегда ----------------------------

    @property
    def paths(self) -> dict[str, Any]:
        return self.data["paths"]

    @property
    def features(self) -> dict[str, Any]:
        return self.data["features"]

    @property
    def quality(self) -> dict[str, Any]:
        return self.data["quality"]

    @property
    def models(self) -> dict[str, Any]:
        return self.data["models"]

    @property
    def validation(self) -> dict[str, Any]:
        return self.data["validation"]

    @property
    def run(self) -> dict[str, Any]:
        return self.data.get("run", {})

    @property
    def pipeline(self) -> dict[str, Any]:
        return self.data.get("pipeline", {})

    @property
    def association(self) -> dict[str, Any]:
        return self.data["quality"]["association"]

    @property
    def quality_rules(self) -> list[dict[str, Any]]:
        return self.data["quality"]["rules"]

    @property
    def feature_cols(self) -> list[str]:
        """Признаки в порядке: сначала числовые, затем категориальные.

        Порядок фиксирован, потому что от него зависит ширина матрицы
        после `ColumnTransformer` и отпечаток схемы (АР-9).
        """
        return list(self.features["numerical"]) + list(self.features["categorical"])

    @property
    def numerical_cols(self) -> list[str]:
        return list(self.features["numerical"])

    @property
    def categorical_cols(self) -> list[str]:
        return list(self.features["categorical"])

    @property
    def target_column(self) -> str:
        return self.data["target"]["column"]

    @property
    def target_name(self) -> str:
        return self.data["target"].get("name", "target")

    @property
    def time_column(self) -> str:
        return self.data["batching"]["time_column"]

    # -- пути ------------------------------------------------------------

    def path(self, key: str) -> Path:
        """Абсолютный путь по ключу из секции `paths`."""
        if key not in self.paths:
            raise ConfigError(f"В секции paths нет ключа {key!r}")
        return self.root / str(self.paths[key])

    @property
    def state_file(self) -> Path:
        return self.path("state_file")

    @property
    def models_dir(self) -> Path:
        return self.path("models")

    @property
    def reports_dir(self) -> Path:
        return self.path("reports")

    @property
    def metadata_dir(self) -> Path:
        return self.path("metadata")

    @property
    def batches_dir(self) -> Path:
        return self.path("data_raw")

    @property
    def rules_file(self) -> Path:
        return self.path("rules_file")

    @property
    def log_file(self) -> Path:
        return self.root / str(self.run.get("log_file", "training.log"))

    @property
    def inference_config(self) -> dict[str, Any]:
        """Настройки устойчивого инференса (6.b.ii)."""
        section = self.get("inference", {})
        if not isinstance(section, dict):
            raise ConfigError("секция inference должна быть словарём")
        return section

    @property
    def inference_threshold_mode(self) -> str:
        """Как выбирается порог принятия решения: `quantile` или `fixed`."""
        mode = str(
            self.inference_config.get("threshold_mode", "quantile")
        ).strip().lower()
        if mode not in ("quantile", "fixed"):
            raise ConfigError(
                f"inference.threshold_mode={mode!r} не поддерживается; "
                f"допустимо: quantile, fixed"
            )
        return mode

    @property
    def inference_fallback_model(self) -> str:
        """Модель, на которую переходим, если продуктовая неприменима."""
        return str(self.inference_config.get("fallback_model", "lr"))

    def target_positive_rate(self, fallback: float = 0.05) -> float:
        """Доля положительных решений, к которой подбирается порог.

        `auto` читается из последних обработанных батчей: доля положи-
        тельного класса падает со временем (с 11 % до 2,4 % за четыре
        года), и фиксированное число устарело бы молча — порог
        подстроился бы под давно неверную долю, и никто бы этого не
        заметил.
        """
        declared = self.inference_config.get("target_positive_rate", "auto")
        if isinstance(declared, (int, float)) and not isinstance(declared, bool):
            value = float(declared)
            if not 0 < value < 1:
                raise ConfigError(
                    f"inference.target_positive_rate={value} вне диапазона (0, 1)"
                )
            return value

        if str(declared).strip().lower() != "auto":
            raise ConfigError(
                f"inference.target_positive_rate={declared!r}: ожидается "
                f"число в (0, 1) или 'auto'"
            )

        for path in sorted(
            self.metadata_dir.glob("quality_*.json"), reverse=True
        )[:self._RATE_LOOKBACK]:
            payload = load_json(path) or {}
            rate = (payload.get("target") or {}).get("positive_rate")
            if isinstance(rate, (int, float)) and 0 < rate < 1:
                return float(rate)
        logger.warning(
            "Доля положительных не найдена в метаданных, берётся %s",
            fallback,
        )
        return float(fallback)

    #: Сколько последних батчей просматривается в поиске доли класса.
    _RATE_LOOKBACK = 5

    @property
    def meta_metric(self) -> str:
        value = str(self.get("meta_learning", {}).get("metric", "f1"))
        if value not in ("f1", "precision", "recall", "roc_auc"):
            raise ConfigError(
                f"meta_learning.metric={value!r} не поддерживается; "
                f"допустимо: f1, precision, recall, roc_auc"
            )
        return value

    @property
    def meta_min_runs(self) -> int:
        """С какого числа батчей выводы Meta Learning считаются обоснованными."""
        try:
            value = int(self.get("meta_learning", {}).get("min_batches", 3))
        except (TypeError, ValueError) as error:
            raise ConfigError(
                f"meta_learning.min_batches должно быть целым числом: {error}"
            ) from error
        if value < 1:
            raise ConfigError("meta_learning.min_batches должно быть не меньше 1")
        return value

    def ensure_dirs(self) -> None:
        """Создать каталоги артефактов. Вызывается один раз оркестратором."""
        for key in ("data_raw", "data_processed", "models", "reports", "metadata"):
            if key in self.paths:
                self.path(key).mkdir(parents=True, exist_ok=True)

    # -- идентичность конфигурации ---------------------------------------

    @property
    def config_hash(self) -> str:
        """Короткий стабильный хэш содержимого конфигурации."""
        payload = json.dumps(self.data, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def fingerprint(self) -> dict[str, Any]:
        """Сведения о конфигурации для манифеста прогона (АР-6)."""
        return {
            "config_hash": self.config_hash,
            "feature_cols": self.feature_cols,
            "numerical_cols": self.numerical_cols,
            "categorical_cols": self.categorical_cols,
            "target": f"{self.target_column} -> {self.target_name}",
            "validation": self.validation,
        }


# ---------------------------------------------------------------------------
# Загрузка и валидация
# ---------------------------------------------------------------------------


def find_project_root(start: Path | None = None) -> Path:
    """Найти корень проекта — каталог, содержащий `config.yaml`.

    Поиск идёт вверх от точки старта (файл модуля или переданный путь).
    Это избавляет от зависимости от текущего рабочего каталога (D-20).
    """
    candidates = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path(__file__).resolve().parent.parent)
    candidates.append(Path.cwd().resolve())

    for candidate in candidates:
        for directory in [candidate, *candidate.parents]:
            if (directory / DEFAULT_CONFIG_NAME).is_file():
                return directory
    raise ConfigError(
        f"Не найден {DEFAULT_CONFIG_NAME}. Запускать нужно из корня проекта "
        f"либо передать путь к конфигурации явно."
    )


def load_config(
    path: str | os.PathLike[str] | None = None,
    root: Path | None = None,
) -> Config:
    """Прочитать и провалидировать конфигурацию.

    Args:
        path: путь к YAML-файлу. Если не задан, ищется `config.yaml`
            по алгоритму `find_project_root`.
        root: корень проекта для разрешения относительных путей.
            По умолчанию — каталог с конфигурацией.
    """
    if path is None:
        root = root or find_project_root()
        path = root / DEFAULT_CONFIG_NAME
    else:
        path = Path(path).resolve()
        if root is None:
            root = path.parent

    if not path.is_file():
        raise ConfigError(f"Файл конфигурации не найден: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ConfigError(f"Ожидался словарь в верхнем уровне {path}, получен {type(raw).__name__}")

    _validate(raw)
    return Config(data=copy.deepcopy(raw), root=Path(root).resolve())


def _validate(cfg: dict[str, Any]) -> None:
    """Проверить структуру конфигурации, накапливая все ошибки.

    Смысл в том, чтобы пользователь увидел сразу все проблемы, а не
    по одной за прогон.
    """
    errors: list[str] = []

    for section in REQUIRED_SECTIONS:
        if section not in cfg:
            errors.append(f"отсутствует обязательная секция {section!r}")
    if errors:
        raise ConfigError("Некорректная конфигурация:\n  - " + "\n  - ".join(errors))

    _validate_paths(cfg, errors)
    _validate_features(cfg, errors)
    _validate_target(cfg, errors)
    _validate_quality(cfg, errors)
    _validate_validation(cfg, errors)
    _validate_models(cfg, errors)

    if errors:
        raise ConfigError("Некорректная конфигурация:\n  - " + "\n  - ".join(errors))


def _validate_paths(cfg: dict[str, Any], errors: list[str]) -> None:
    required = (
        "data_raw", "models", "reports", "metadata",
        "state_file", "rules_file",
    )
    for key in required:
        if key not in cfg["paths"]:
            errors.append(f"paths.{key} отсутствует")
        elif not str(cfg["paths"][key]).strip():
            errors.append(f"paths.{key} пуст")


def _validate_features(cfg: dict[str, Any], errors: list[str]) -> None:
    features = cfg["features"]
    for key in ("numerical", "categorical"):
        if key not in features:
            errors.append(f"features.{key} отсутствует")
            return
        if not isinstance(features[key], list) or not features[key]:
            errors.append(f"features.{key} должен быть непустым списком")

    numerical = set(features.get("numerical", []))
    categorical = set(features.get("categorical", []))
    overlap = numerical & categorical
    if overlap:
        errors.append(f"признак указан и как числовой, и как категориальный: {sorted(overlap)}")


def _validate_target(cfg: dict[str, Any], errors: list[str]) -> None:
    target = cfg["target"]
    if "column" not in target:
        errors.append("target.column отсутствует")

    features = set(cfg["features"].get("numerical", [])) | set(cfg["features"].get("categorical", []))
    if target.get("column") in features:
        errors.append(
            f"target.column ({target['column']!r}) не должен одновременно быть признаком"
        )

    rule = target.get("positive_rule", "notna")
    if rule not in ("notna", "positive"):
        errors.append(f"target.positive_rule должен быть 'notna' или 'positive', получено {rule!r}")


def _validate_quality(cfg: dict[str, Any], errors: list[str]) -> None:
    quality = cfg["quality"]

    for key in ("rules", "association"):
        if key not in quality:
            errors.append(f"quality.{key} отсутствует")
            return

    for key in ("max_row_missing_ratio", "max_violation_ratio"):
        value = quality.get(key)
        if value is None:
            errors.append(f"quality.{key} отсутствует")
        elif not 0.0 <= float(value) <= 1.0:
            errors.append(f"quality.{key} должен быть в [0, 1], получено {value!r}")

    seen: set[str] = set()
    for index, rule in enumerate(quality["rules"]):
        label = f"quality.rules[{index}]"
        name = rule.get("name")
        if not name:
            errors.append(f"{label}: отсутствует name")
        elif name in seen:
            errors.append(f"{label}: дублирующееся имя {name!r}")
        else:
            seen.add(name)

        severity = rule.get("severity", "error")
        if severity not in SEVERITIES:
            errors.append(f"{label} ({name}): severity должен быть одним из {SEVERITIES}")

        on_missing = rule.get("on_missing", "ignore")
        if on_missing not in ("ignore", "violation"):
            errors.append(
                f"{label} ({name}): on_missing должен быть 'ignore' или 'violation'"
            )

        _validate_rule_condition(rule, label, name, errors, allow_conditional=True)

    assoc = quality["association"]
    for key in ("min_support", "min_confidence"):
        value = assoc.get(key)
        if value is None or not 0.0 <= float(value) <= 1.0:
            errors.append(f"quality.association.{key} должен быть в [0, 1], получено {value!r}")
    n_rules = assoc.get("n_rules")
    if not isinstance(n_rules, int) or n_rules < 1:
        errors.append("quality.association.n_rules должен быть целым >= 1")
    max_len = assoc.get("max_len")
    if max_len is not None and (not isinstance(max_len, int) or max_len < 1):
        errors.append("quality.association.max_len должен быть целым >= 1 или null")
    min_antecedent = assoc.get("min_antecedent_rows", 30)
    if not isinstance(min_antecedent, int) or min_antecedent < 1:
        errors.append("quality.association.min_antecedent_rows должен быть целым >= 1")
    algorithm = assoc.get("algorithm", "fpgrowth")
    if algorithm not in ("fpgrowth", "apriori"):
        errors.append(
            f"quality.association.algorithm должен быть 'fpgrowth' или 'apriori', получено {algorithm!r}"
        )


def _validate_rule_condition(
    rule: dict[str, Any],
    label: str,
    name: str | None,
    errors: list[str],
    allow_conditional: bool,
) -> None:
    """Проверить тело правила.

    Поддерживаются две формы: одиночное условие (`column`/`op`/`value`)
    и условное правило (`applies_when` + одиночное условие) — например
    «грузоподъёмность должна быть > 0, если тип кузова грузовой».
    """
    if allow_conditional and "applies_when" in rule:
        applies = rule["applies_when"]
        if not isinstance(applies, dict):
            errors.append(f"{label} ({name}): applies_when должен быть словарём")
        else:
            _validate_condition(applies, f"{label} ({name}) applies_when", errors)

    column = rule.get("column")
    if not column:
        errors.append(f"{label} ({name}): отсутствует column")
        return

    op = rule.get("op")
    if op not in VALID_OPERATORS:
        errors.append(
            f"{label} ({name}): op={op!r} недопустим; допустимые: {', '.join(VALID_OPERATORS)}"
        )
        return

    if op in ("between", "not_between"):
        bounds = rule.get("value")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            errors.append(f"{label} ({name}): op={op} требует value из двух границ")
        elif float(bounds[0]) >= float(bounds[1]):
            errors.append(f"{label} ({name}): нижняя граница должна быть меньше верхней")
    elif op in ("in", "not_in"):
        if not isinstance(rule.get("value"), (list, tuple)) or not rule["value"]:
            errors.append(f"{label} ({name}): op={op} требует непустой список value")
    elif op not in ("notna", "isna", "positive", "non_negative"):
        if "value" not in rule:
            errors.append(f"{label} ({name}): op={op} требует value")


def _validate_condition(condition: dict[str, Any], label: str, errors: list[str]) -> None:
    """Проверить простое условие без вложенности."""
    if not condition.get("column"):
        errors.append(f"{label}: отсутствует column")
    op = condition.get("op")
    if op not in VALID_OPERATORS:
        errors.append(f"{label}: op={op!r} недопустим")
    elif op in ("in", "not_in") and not condition.get("value"):
        errors.append(f"{label}: op={op} требует непустой список value")
    elif op in ("between", "not_between"):
        bounds = condition.get("value")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            errors.append(f"{label}: op={op} требует две границы")


def _validate_validation(cfg: dict[str, Any], errors: list[str]) -> None:
    validation = cfg["validation"]
    strategy = validation.get("strategy", "stream_holdout")
    if strategy not in ("stream_holdout", "holdout"):
        errors.append(
            f"validation.strategy должен быть 'stream_holdout' или 'holdout', получено {strategy!r}"
        )
    test_size = validation.get("test_size")
    if test_size is None or not 0.0 < float(test_size) < 1.0:
        errors.append("validation.test_size должен быть в (0, 1)")


def _validate_models(cfg: dict[str, Any], errors: list[str]) -> None:
    models = cfg["models"]
    for name in ("mlp", "dt"):
        if name not in models:
            errors.append(f"models.{name} отсутствует")
            continue
        if not isinstance(models[name], dict):
            errors.append(f"models.{name} должен быть словарём с гиперпараметрами")

    mlp = models.get("mlp", {})
    if mlp:
        layers = mlp.get("hidden_layer_sizes")
        if not isinstance(layers, (list, tuple)) or not layers:
            errors.append("models.mlp.hidden_layer_sizes должен быть непустым списком")
        elif mlp.get("partial_fit") and mlp.get("warm_start", True):
            # partial_fit игнорирует warm_start: повторный fit перезапустит
            # оптимизацию с нуля. Молча оставлять оба флага — ловушка.
            errors.append(
                "models.mlp: нельзя одновременно partial_fit=true и warm_start=true; "
                "для инкрементального обучения warm_start должен быть false"
            )
