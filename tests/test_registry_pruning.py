"""Проверки ограничения роста каталога моделей.

Каждый батч создаёт четыре версии, и без очистки на 48 батчах каталог
`models/` занимает около 160 МБ — почти всё это случайный лес. В CI
такой каталог уезжает и в кэш состояния, и в артефакты при каждом
запуске. Проверяется, что очистка удаляет только файлы, не трогает
продуктовую версию и не выбрасывает историю метрик.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Config
from src.registry import ModelRegistry, QualityGate


class FakeModel:
    """Объект вместо обученной модели: pickle достаточно."""

    def __init__(self, size: int = 16) -> None:
        self.payload = b"x" * size


def permissive_gate(**overrides) -> QualityGate:
    """Гейт, пропускающий любого кандидата.

    Иначе продуктовая версия не появлялась бы и проверять её защиту
    было бы не на чем.
    """
    rules = {
        "min_f1": 0.0, "min_roc_auc": 0.0, "max_regression_f1": 1.0,
        "max_psi": 1.0, "promote_best_by_f1": True,
    }
    rules.update(overrides)
    return QualityGate(rules)


def make_registry(tmp_path: Path, **gate) -> ModelRegistry:
    return ModelRegistry(tmp_path, gate=permissive_gate(**gate))


def register(registry: ModelRegistry, model_name: str, batch: int, f1: float) -> dict:
    # `roc_auc` обязателен: гейт отклоняет кандидата с отсутствующей
    # метрикой, и без неё продуктовая версия просто не появилась бы.
    return registry.register(
        FakeModel(), model_name, batch, f"batch_{batch:04d}",
        {"f1": f1, "roc_auc": 0.7},
    )


def improving(registry: ModelRegistry, model: str = "lr", n: int = 6) -> None:
    """Зарегистрировать `n` версий с растущим f1.

    Растущий f1 нужен, чтобы продуктовой стала последняя версия: иначе
    продуктовая застревает на первой и служит третьим исключением из
    правила «остаются последние `keep`», о чём есть отдельная проверка.
    """
    for batch in range(n):
        register(registry, model, batch, 0.1 + batch * 0.01)


@pytest.fixture
def registry(tmp_path) -> ModelRegistry:
    return make_registry(tmp_path)


# ---------------------------------------------------------------------------
# Базовая очистка
# ---------------------------------------------------------------------------


def test_old_artifacts_are_removed(registry):
    """За батч создаётся четыре версии; старые файлы исчезают."""
    for batch in range(8):
        for name in ("lr", "dt", "rf", "mlp"):
            register(registry, name, batch, 0.1 + batch * 0.01)

    before = len(list(registry.models_dir.glob("*__v*.pkl")))
    result = registry.prune(keep=2)
    after = len(list(registry.models_dir.glob("*__v*.pkl")))

    assert before == 32
    # По две последние версии каждой из четырёх моделей
    assert after == 8
    assert result["freed_bytes"] > 0
    assert len(result["removed"]) == 24


def test_registry_keeps_every_version(registry):
    """Очистка файлов не выбрасывает историю: записи реестра все на месте.

    Иначе исчезла бы возможность посмотреть, как менялось качество
    по батчам, — а это и есть смысл реестра.
    """
    for batch in range(8):
        for name in ("lr", "dt", "rf", "mlp"):
            register(registry, name, batch, 0.1 + batch * 0.01)

    registry.prune(keep=2)

    assert len(registry.versions) == 32
    assert registry.summary()["n_versions"] == 32
    metrics = [
        entry["metrics"]["f1"]
        for entry in registry.versions
        if entry["model"] == "lr"
    ]
    assert len(metrics) == 8


def test_pruned_entries_are_marked(registry):
    """Запись без файла помечена, чтобы отчёт не врал о наличии артефакта."""
    improving(registry, n=6)

    registry.prune(keep=2)

    flagged = [e for e in registry.versions if e.get("artifact_pruned")]
    kept = [e for e in registry.versions if not e.get("artifact_pruned")]
    assert len(flagged) == 4
    assert len(kept) == 2
    # Метрики у помеченных записей на месте
    assert all("metrics" in entry for entry in flagged)


def test_pruned_count_is_reported(registry):
    """Сводка реестра показывает, сколько версий осталось без файла."""
    improving(registry, n=6)

    registry.prune(keep=1)

    summary = registry.summary()
    assert summary["n_versions"] == 6
    assert summary["n_artifacts_pruned"] == 5


# ---------------------------------------------------------------------------
# Что удалять нельзя
# ---------------------------------------------------------------------------


def test_production_artifact_is_never_removed(tmp_path):
    """Продуктовая версия переживает очистку, даже если она старейшая.

    На неё ссылается `best_model.pkl` и инференс. Если продуктовой
    назначена версия первого батча, а лимит равен одному, такой файл
    всё равно должен остаться на диске.
    """
    registry = make_registry(tmp_path)
    for batch in range(5):
        for name in ("lr", "mlp"):
            register(registry, name, batch, 0.30 if batch == 0 else 0.1)

    production = registry.production
    assert production is not None
    assert production["batch_idx"] == 0, "подготовка неудачна: нужен старый продукт"

    registry.prune(keep=1)

    artifact = registry.models_dir / production["artifact"]
    assert artifact.is_file()
    assert not production.get("artifact_pruned")


def test_pointers_survive_pruning(registry):
    """Указатели `best_model.pkl` и `<model>_latest.pkl` остаются рабочими.

    Инференс читает именно их, и они не должны быть удалены даже при
    агрессивном лимите.
    """
    for batch in range(5):
        for name in ("lr", "rf"):
            register(registry, name, batch, 0.1 + batch * 0.01)

    registry.refresh_pointers()
    registry.prune(keep=1)

    for name in ("best_model.pkl", "lr_latest.pkl", "rf_latest.pkl"):
        assert (registry.models_dir / name).is_file(), name


def test_zero_disables_pruning(registry):
    """Нулевой лимит выключает очистку, а не удаляет всё."""
    for batch in range(4):
        register(registry, "lr", batch, 0.1)

    result = registry.prune(keep=0)

    assert result["removed"] == []
    assert len(list(registry.models_dir.glob("*__v*.pkl"))) == 4


def test_negative_limit_disables_pruning(registry):
    """Отрицательное значение не должно приводить к удалению всего."""
    for batch in range(4):
        register(registry, "lr", batch, 0.1)

    result = registry.prune(keep=-3)

    assert result["removed"] == []
    assert len(list(registry.models_dir.glob("*__v*.pkl"))) == 4


# ---------------------------------------------------------------------------
# Повторные и граничные вызовы
# ---------------------------------------------------------------------------


def test_prune_is_idempotent(registry):
    """Повторный вызов ничего не делает и не падает на уже удалённом.

    Иначе каждый батч сообщал бы об одних и тех же удалениях, и в логе
    очистка выглядела бы как работа, которой не было.
    """
    improving(registry, n=6)

    first = registry.prune(keep=2)
    second = registry.prune(keep=2)

    assert first["removed"]
    assert second["removed"] == []


def test_prune_with_fewer_versions_than_limit(registry):
    """Пока версий меньше лимита, удалять нечего."""
    register(registry, "lr", 0, 0.1)
    register(registry, "lr", 1, 0.1)

    result = registry.prune(keep=5)

    assert result["removed"] == []
    assert len(list(registry.models_dir.glob("*__v*.pkl"))) == 2


def test_empty_registry_is_pruned_safely(tmp_path):
    """Пустой реестр не приводит к исключению."""
    result = make_registry(tmp_path).prune(keep=3)
    assert result == {"removed": [], "freed_bytes": 0, "pruned_entries": 0}


# ---------------------------------------------------------------------------
# Связка с конфигурацией
# ---------------------------------------------------------------------------


def test_pipeline_reads_limit_from_config(tmp_path):
    """Лимит берётся из конфигурации, а не зашит в код."""
    from src.pipeline import Pipeline

    config = Config(
        data={
            "paths": {
                "data_raw": "raw", "models": "models", "reports": "reports",
                "metadata": "metadata", "state_file": "state.json",
                "rules_file": "rules.json", "data_processed": "processed",
            },
            "registry": {"keep_artifacts_per_model": 3},
        },
        root=tmp_path,
    )
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.config = config
    pipeline.registry = ModelRegistry(tmp_path / "models", gate=permissive_gate())

    for batch in range(6):
        register(pipeline.registry, "lr", batch, 0.1 + batch * 0.01)

    pipeline._prune_models()

    assert len(list(pipeline.registry.models_dir.glob("*__v*.pkl"))) == 3


def test_default_limit_applies_without_section():
    """Без секции `registry` работает значение по умолчанию — 5."""
    from src.config import load_config

    config = load_config()
    assert config.get("registry", {}).get("keep_artifacts_per_model") == 5
