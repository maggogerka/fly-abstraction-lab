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


@pytest.mark.parametrize(
    ("profile", "device"),
    [
        ("smoke_cpu", "cpu"),
        ("smoke_gpu", "cuda"),
        ("pilot_gpu", "cuda"),
        ("paper_gpu", "cuda"),
    ],
)
def test_readiness_profiles_are_valid(profile: str, device: str) -> None:
    config = load_config(profile)
    assert config["runtime"]["device"] == device
    assert config["task"]["mode"] == "numeric"


def test_pilot_profile_fits_tiny_graph_generator_and_uses_geometry_ood() -> None:
    config = load_config("pilot_gpu")
    assert config["graph"]["max_graph_nodes"] == 128
    assert config["graph"]["expected_num_nodes"] == 128
    assert config["model"]["hidden_size"] == 128
    assert config["data"]["split_strategy"] == "held_out_geometry"
    assert config["task"]["primary_split"] == "test_held_out_geometry_ood"


def test_paper_profile_requires_resource_acceptance() -> None:
    config = load_config("paper_gpu")
    assert config["resource_guard"]["require_explicit_acceptance"] is True
