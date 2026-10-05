import pytest

from fly_abstraction.config import load_config
from fly_abstraction.resources import ResourceGuardError, enforce_resource_guard, estimate_resources


def test_resource_estimate_reports_requested_shape() -> None:
    config = load_config("smoke_cpu")
    estimate = estimate_resources(
        config,
        available_ram_bytes=8 * 1024**3,
        available_vram_bytes=0,
    )
    assert estimate.nodes == 16
    assert estimate.edges == 32
    assert estimate.batch_size == 2
    assert estimate.sequence_length == 32
    assert estimate.estimated_vram_bytes > 0
    enforce_resource_guard(config, estimate)


def test_resource_guard_rejects_dangerous_parameters() -> None:
    config = load_config("smoke_cpu")
    config["graph"]["expected_num_nodes"] = 2_000_000
    config["graph"]["expected_num_edges"] = 100_000_000
    estimate = estimate_resources(config, available_ram_bytes=1024**3, available_vram_bytes=0)
    with pytest.raises(ResourceGuardError, match="Reduce nodes, edges, batch size"):
        enforce_resource_guard(config, estimate)


def test_pilot_estimate_fits_12_gib_rtx_5070_headroom() -> None:
    config = load_config("pilot_gpu")
    estimate = estimate_resources(
        config,
        available_ram_bytes=32 * 1024**3,
        available_vram_bytes=12 * 1024**3,
    )
    enforce_resource_guard(config, estimate)
    assert estimate.estimated_vram_bytes < 12 * 1024**3 * 0.8
