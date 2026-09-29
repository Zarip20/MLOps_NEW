"""Проверки раздела реестра в текстовом отчёте.

Раздел обязан быть честным про устаревшие файлы моделей. Молчание
выглядело бы как «история потеряна», хотя на деле удалены только
файлы, а все метрики остались в реестре.
"""

from __future__ import annotations

import pytest

from src.report import _registry_lines


def manifest(**registry) -> dict:
    return {"registry": registry}


BASE = {
    "n_versions": 192,
    "by_model": {"lr": 48, "dt": 48, "rf": 48, "mlp": 48},
    "n_artifacts_pruned": 171,
    "production": {
        "version": "v0140", "model": "mlp", "batch_idx": 34, "f1": 0.3495,
    },
    "rejected": 50,
    "last_rejection": None,
    "versions": [
        {"version": "v0140", "model": "mlp", "batch_idx": 34,
         "f1": 0.3495, "status": "production", "reasons": []},
    ],
}


def test_pruned_files_are_reported():
    """Отчёт говорит, сколько файлов удалено и что метрики сохранены."""
    text = "\n".join(_registry_lines([manifest(**BASE)]))

    assert "Всего версий: 192" in text
    assert "21 хранятся" in text
    assert "171 удалены как устаревшие" in text
    # Читатель не должен решить, что история потеряна
    assert "метрики всех версий сохранены" in text


def test_count_comes_from_registry_not_from_trimmed_list():
    """Счётчик берётся из реестра, а не из урезанного списка версий.

    В манифесте записи версий содержат только ключевые поля, признака
    очистки среди них нет. Раньше отчёт считал их по этому списку и
    показывал ноль, хотя 171 файл был удалён.
    """
    data = dict(BASE)
    data["versions"] = [
        {"version": "v0001", "model": "lr", "batch_idx": 0,
         "f1": 0.1, "status": "previous", "reasons": []},
    ]
    text = "\n".join(_registry_lines([manifest(**data)]))

    assert "171 удалены как устаревшие" in text


def test_nothing_is_said_when_no_pruning():
    """Если очистка не выполнялась, лишней строки на отчёте нет."""
    data = dict(BASE)
    data["n_artifacts_pruned"] = 0
    text = "\n".join(_registry_lines([manifest(**data)]))

    assert "устаревшие" not in text
    assert "Всего версий: 192" in text


def test_missing_counter_is_tolerated():
    """Манифест без счётчика не приводит к исключению."""
    data = {key: value for key, value in BASE.items() if key != "n_artifacts_pruned"}
    text = "\n".join(_registry_lines([manifest(**data)]))

    assert "устаревшие" not in text
    assert "Всего версий: 192" in text


def test_inconsistent_counters_are_reported_not_hidden():
    """Разошедшиеся счётчики видны, а не превращаются в «−170 хранится».

    Молчаливое отрицательное или нулевое число выглядело бы как ошибка
    очистки, и читатель отчёта принял бы её за факт.
    """
    data = dict(BASE)
    data["n_versions"] = 5
    data["n_artifacts_pruned"] = 171
    text = "\n".join(_registry_lines([manifest(**data)]))

    assert "ВНИМАНИЕ" in text
    assert "требует проверки" in text
    assert "хранятся" not in text


def test_production_version_is_named():
    """Продуктовая версия указана явно: её ищут первой."""
    text = "\n".join(_registry_lines([manifest(**BASE)]))
    assert "Продуктовая: mlp v0140" in text


def test_empty_registry_is_handled():
    """Пустой реестр не приводит к исключению."""
    text = "\n".join(_registry_lines([{"registry": {"versions": []}}]))
    assert "Реестр пуст" in text


def test_versions_without_manifest_are_handled():
    """Отсутствие манифестов не приводит к исключению."""
    assert _registry_lines([]) is not None
