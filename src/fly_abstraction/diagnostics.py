"""Read-only runtime diagnostics and a single-step GPU smoke check."""

from __future__ import annotations

import math
import platform
import sys
from typing import Any

import torch

from fly_abstraction import __version__
from fly_abstraction.graph.connectome import tiny_synthetic_graph
from fly_abstraction.models.connectome import TrainableConnectomeRNN


def gpu_report() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    compiled_arches = torch.cuda.get_arch_list()
    report: dict[str, Any] = {
        "package_version": __version__,
        "python": platform.python_version(),
        "python_supported": sys.version_info[:2] == (3, 11),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "compiled_arches": compiled_arches,
        "sm_120_compiled": "sm_120" in compiled_arches,
        "cuda_12_8_or_newer": False,
        "gpu": None,
        "compute_capability": None,
        "vram_total_bytes": None,
        "mixed_precision": {"fp16": False, "bf16": False, "tf32": False},
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        capability = torch.cuda.get_device_capability(0)
        report.update(
            {
                "gpu": properties.name,
                "compute_capability": f"{capability[0]}.{capability[1]}",
                "vram_total_bytes": int(properties.total_memory),
                "vram_total_gib": round(properties.total_memory / 1024**3, 3),
                "mixed_precision": {
                    "fp16": True,
                    "bf16": bool(torch.cuda.is_bf16_supported()),
                    "tf32": capability[0] >= 8,
                },
            }
        )
    cuda_parts = tuple(int(part) for part in str(torch.version.cuda or "0.0").split(".")[:2])
    report["cuda_12_8_or_newer"] = cuda_parts >= (12, 8)
    capability_parts = tuple(
        int(part) for part in str(report["compute_capability"] or "0.0").split(".")
    )
    report["blackwell_ready"] = bool(
        cuda_available
        and report["sm_120_compiled"]
        and report["cuda_12_8_or_newer"]
        and capability_parts >= (12, 0)
    )
    return report


def run_gpu_smoke(seed: int = 17) -> dict[str, Any]:
    """Run one tiny forward/backward on CUDA; no optimizer and no parameter update."""
    report = gpu_report()
    if not report["blackwell_ready"]:
        raise RuntimeError(
            "GPU smoke requires CUDA 12.8+, a visible CUDA GPU, and PyTorch compiled for sm_120"
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    graph = tiny_synthetic_graph(8, seed).normalized().to(device)
    model = TrainableConnectomeRNN(
        graph, vocab_size=32, embedding_dim=4, expression_vocab_size=32
    ).to(device)
    token_ids = torch.randint(0, 32, (2, 8), device=device)
    numeric = torch.zeros(2, 8, 1, device=device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype):
        output = model(token_ids, numeric)
        loss = output.answer.float().square().mean()
    loss.backward()
    gradient = model.cell.edge_scale.grad
    finite = bool(math.isfinite(float(loss.detach())) and gradient is not None)
    finite = finite and bool(torch.isfinite(gradient).all())
    if not finite:
        raise FloatingPointError("GPU smoke produced a non-finite loss or gradient")
    return {
        **report,
        "smoke": {
            "forward": "ok",
            "backward": "ok",
            "optimizer_step": False,
            "autocast_dtype": str(dtype),
            "loss_finite": True,
            "gradient_finite": True,
        },
    }
