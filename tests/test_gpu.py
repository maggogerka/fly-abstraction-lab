import pytest

from fly_abstraction.diagnostics import gpu_report, run_gpu_smoke


@pytest.mark.skipif(
    not gpu_report()["blackwell_ready"], reason="Blackwell CUDA 12.8 runtime is unavailable"
)
def test_tiny_gpu_forward_backward_without_optimizer() -> None:
    report = run_gpu_smoke()
    assert report["smoke"]["optimizer_step"] is False
    assert report["smoke"]["gradient_finite"] is True
