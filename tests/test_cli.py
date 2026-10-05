import argparse
from pathlib import Path

import pytest

from fly_abstraction.cli import _relative_path, main
from fly_abstraction.data.adapters import TinyDatasetAdapter


def test_train_defaults_to_side_effect_free_dry_run(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["--profile", "local_cpu", "train"]) == 0
    assert "DRY-RUN COMPLETE" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()


def test_tiny_prepare_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "tiny.jsonl"
    TinyDatasetAdapter.prepare(path, count=10, seed=17)
    with pytest.raises(FileExistsError):
        TinyDatasetAdapter.prepare(path, count=10, seed=17)


def test_evaluate_without_predictions_is_a_dry_run(capsys) -> None:
    assert main(["evaluate"]) == 0
    assert "DRY-RUN" in capsys.readouterr().out


def test_doctor_and_gpu_doctor_are_read_only(capsys) -> None:
    assert main(["doctor"]) == 0
    assert '"python"' in capsys.readouterr().out
    result = main(["doctor-gpu"])
    assert result in {0, 1}
    assert '"sm_120_compiled"' in capsys.readouterr().out


def test_registered_download_is_dry_without_confirmation(capsys) -> None:
    assert main(["data", "download", "uci_energy_efficiency"]) == 0
    assert "no download performed" in capsys.readouterr().out


def test_cli_paths_cannot_escape_repository() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _relative_path("../outside")


def test_windows_friend_entrypoints_keep_dangerous_actions_explicit() -> None:
    root = Path(__file__).resolve().parents[1]
    cmd = (root / "START_HERE.cmd").read_text(encoding="utf-8")
    script = (root / "scripts" / "setup_friend_pc.ps1").read_text(encoding="utf-8-sig")
    assert "setup_friend_pc.ps1" in cmd
    assert cmd.isascii()
    assert "%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" in cmd
    assert "chcp" not in cmd.lower()
    assert "TRAIN PILOT" in script
    assert "DOWNLOAD UCI" in script
    assert "Find-NvidiaSmi" in script
    assert "$gpuOutput = @(& $nvidiaSmi" in script
    assert "$gpuExitCode = $LASTEXITCODE" in script
    assert script.index("$gpuExitCode = $LASTEXITCODE") < script.index(
        "$gpuLine = $gpuOutput | Select-Object -First 1"
    )
    assert "if ($gpuExitCode -ne 0)" in script
    assert "$previousErrorActionPreference = $ErrorActionPreference" in script
    assert "$ErrorActionPreference = 'Continue'" in script
    assert "$ErrorActionPreference = $previousErrorActionPreference" in script
    assert "if ($exitCode -ne 0)" in script
    assert "RTX 5090" not in script
    assert "--confirm-download" in script
    assert "--confirm-train" in script
    assert "pip install" not in script
    assert "CUDA Toolkit" in script
