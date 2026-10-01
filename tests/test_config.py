from pathlib import Path

import pytest

from fly_abstraction.config import ConfigError, load_config, local_safety_violations


def test_local_profile_has_required_hard_limits() -> None:
    config = load_config("local_cpu")
    assert config["data"]["max_examples"] == 500
    assert config["training"]["max_epochs"] == 1
    assert config["graph"]["max_graph_nodes"] == 128
    assert config["training"]["batch_size"] == 4
    assert config["training"]["num_workers"] == 0
    assert config["runtime"]["torch_threads"] == 2
    assert config["runtime"]["mixed_precision"] is False
    assert config["training"]["checkpoints"] is False
    assert local_safety_violations(config) == []


def test_override_violation_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.yaml"
    path.write_text("training:\n  max_epochs: 2\n", encoding="utf-8")
    config = load_config("local_cpu", path)
    assert "training.max_epochs=2 exceeds 1" in local_safety_violations(config)


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ConfigError):
        load_config("missing")
