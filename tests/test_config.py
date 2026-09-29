"""Проверки загрузки и валидации конфигурации (АР-1).

Конфигурация — единственный источник истины, поэтому её проверка должна
 catching типовые ошибки заранее, а не в середине прогона.
"""

from __future__ import annotations

import pytest
import yaml

from src.config import ConfigError, load_config


BASE = {
    "paths": {
        "data_raw": "data/raw_batches",
        "models": "models",
        "reports": "reports",
        "metadata": "data/metadata",
        "state_file": "state.json",
        "rules_file": "data/rules.json",
    },
    "batching": {"size": "month", "time_column": "INSR_BEGIN"},
    "features": {"numerical": ["A"], "categorical": ["B"]},
    "target": {"column": "Y", "name": "HAS_Y", "positive_rule": "notna"},
    "quality": {
        "max_row_missing_ratio": 0.25,
        "max_violation_ratio": 0.1,
        "rules": [{"name": "r1", "column": "A", "op": "ge", "value": 0}],
        "association": {
            "algorithm": "fpgrowth",
            "min_support": 0.01,
            "min_confidence": 0.5,
            "min_lift": 1.2,
            "max_len": 2,
            "n_rules": 5,
            "min_antecedent_rows": 30,
        },
    },
    "preprocessing": {},
    "validation": {"strategy": "stream_holdout", "test_size": 0.2},
    "models": {"mlp": {"hidden_layer_sizes": [4]}, "dt": {"max_depth": 3}},
}


def write_config(tmp_path, payload) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return str(path)


def test_valid_config_loads(tmp_path):
    config = load_config(write_config(tmp_path, BASE))
    assert config.feature_cols == ["A", "B"]
    assert config.target_name == "HAS_Y"
    assert config.time_column == "INSR_BEGIN"
    assert len(config.config_hash) == 12


def test_missing_section_is_reported(tmp_path):
    payload = {key: value for key, value in BASE.items() if key != "validation"}
    with pytest.raises(ConfigError) as error:
        load_config(write_config(tmp_path, payload))
    assert "validation" in str(error.value)


def test_all_errors_reported_at_once(tmp_path):
    """Пользователь видит сразу все проблемы, а не по одной за запуск."""
    payload = dict(BASE)
    payload["quality"] = dict(BASE["quality"], max_row_missing_ratio=5.0)
    payload["validation"] = dict(BASE["validation"], test_size=1.5)
    with pytest.raises(ConfigError) as error:
        load_config(write_config(tmp_path, payload))
    message = str(error.value)
    assert "max_row_missing_ratio" in message
    assert "test_size" in message


def test_partial_fit_with_warm_start_is_rejected(tmp_path):
    """Ловушка сочетания partial_fit и warm_start отвергается.

    partial_fit не переиспользует прогресс оптимизатора, поэтому
    warm_start=true рядом с partial_fit=true вводит в заблуждение:
    кажется, что оптимизация продолжается, а она начинается заново.
    """
    payload = dict(BASE)
    payload["models"] = {
        "mlp": {"hidden_layer_sizes": [4], "partial_fit": True, "warm_start": True},
        "dt": {"max_depth": 3},
    }
    with pytest.raises(ConfigError) as error:
        load_config(write_config(tmp_path, payload))
    assert "partial_fit" in str(error.value)


def test_unknown_operator_is_rejected(tmp_path):
    payload = dict(BASE)
    payload["quality"] = dict(
        BASE["quality"],
        rules=[{"name": "r1", "column": "A", "op": "не_оператор", "value": 0}],
    )
    with pytest.raises(ConfigError) as error:
        load_config(write_config(tmp_path, payload))
    assert "не_оператор" in str(error.value)


def test_duplicate_rule_names_are_rejected(tmp_path):
    payload = dict(BASE)
    payload["quality"] = dict(
        BASE["quality"],
        rules=[
            {"name": "same", "column": "A", "op": "ge", "value": 0},
            {"name": "same", "column": "A", "op": "le", "value": 5},
        ],
    )
    with pytest.raises(ConfigError) as error:
        load_config(write_config(tmp_path, payload))
    assert "дублирующееся" in str(error.value)


def test_invalid_on_missing_is_rejected(tmp_path):
    payload = dict(BASE)
    payload["quality"] = dict(
        BASE["quality"],
        rules=[{"name": "r1", "column": "A", "op": "ge", "value": 0, "on_missing": "maybe"}],
    )
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, payload))


def test_between_requires_two_bounds(tmp_path):
    payload = dict(BASE)
    payload["quality"] = dict(
        BASE["quality"],
        rules=[{"name": "r1", "column": "A", "op": "between", "value": [1, 2, 3]}],
    )
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, payload))


def test_feature_cannot_be_both_numeric_and_categorical(tmp_path):
    payload = dict(BASE)
    payload["features"] = {"numerical": ["A", "B"], "categorical": ["B"]}
    with pytest.raises(ConfigError) as error:
        load_config(write_config(tmp_path, payload))
    assert "B" in str(error.value)


def test_target_cannot_be_a_feature(tmp_path):
    payload = dict(BASE)
    payload["features"] = {"numerical": ["A"], "categorical": ["Y"]}
    with pytest.raises(ConfigError) as error:
        load_config(write_config(tmp_path, payload))
    assert "target.column" in str(error.value)


def test_paths_resolve_against_config_root(tmp_path):
    config = load_config(write_config(tmp_path, BASE))
    assert config.models_dir == tmp_path / "models"
    assert config.batches_dir == tmp_path / "data" / "raw_batches"
    assert config.state_file == tmp_path / "state.json"


def test_config_hash_is_stable_and_sensitive(tmp_path):
    first = load_config(write_config(tmp_path, BASE))
    second = load_config(write_config(tmp_path, BASE))
    assert first.config_hash == second.config_hash

    changed = dict(BASE)
    changed["models"] = {"mlp": {"hidden_layer_sizes": [8]}, "dt": {"max_depth": 5}}
    third = load_config(write_config(tmp_path, changed))
    assert third.config_hash != first.config_hash


def test_project_config_is_valid():
    """Боевой config.yaml должен проходить собственную проверку."""
    config = load_config()
    assert len(config.quality_rules) == 5
    assert config.association["algorithm"] in ("fpgrowth", "apriori")
    assert config.validation["test_size"] == 0.2
    # Ключи, которые прежний код игнорировал, теперь обязательны
    assert config.path("data_raw")
    assert config.paths["state_file"]
