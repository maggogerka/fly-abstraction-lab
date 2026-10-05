import pytest
import torch

from fly_abstraction.diagnostics import gpu_report, run_gpu_smoke


def test_gpu_report_accepts_rtx_5070_blackwell(monkeypatch) -> None:
    class Properties:
        name = "NVIDIA GeForce RTX 5070"
        total_memory = 12 * 1024**3

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_120"])
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: Properties())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (12, 0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    report = gpu_report()
    assert report["gpu"] == "NVIDIA GeForce RTX 5070"
    assert report["vram_total_gib"] == 12.0
    assert report["blackwell_ready"] is True


@pytest.mark.skipif(
    not gpu_report()["blackwell_ready"], reason="Blackwell CUDA 12.8 runtime is unavailable"
)
def test_tiny_gpu_forward_backward_without_optimizer() -> None:
    report = run_gpu_smoke()
    assert report["smoke"]["optimizer_step"] is False
    assert report["smoke"]["gradient_finite"] is True
