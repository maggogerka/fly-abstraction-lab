# RTX 5070/50-series research-PC workflow

This workflow targets a clean 64-bit Windows 10/11 PC with RTX 5070 (or another Blackwell
RTX 50-series GPU) and 32 GB RAM.
The host needs only an up-to-date NVIDIA driver, WSL2, Git (unless using a ZIP), and
Docker Desktop with the WSL2 engine. Do **not** install host Python, PyTorch, CUDA Toolkit,
or cuDNN: Python 3.11, PyTorch 2.7.1, CUDA 12.8, and cuDNN 9 are pinned inside Docker.

No command in the setup/doctor/smoke path downloads datasets or trains a model.

## Recommended Windows path

Clone the branch or unpack its ZIP:

```text
git clone --branch feat/research-pc-readiness https://github.com/maggogerka/fly-abstraction-lab.git
cd fly-abstraction-lab
```

Double-click `START_HERE.cmd`. The guided action:

1. checks Windows x64 and warns below 25 GB free disk;
2. checks `nvidia-smi`, driver version, GPU and VRAM;
3. checks WSL2, Git, Docker Desktop, Compose, and the Docker engine;
4. offers missing Git/Docker via `winget` only after an exact confirmation;
5. builds the digest-pinned `research-gpu` image;
6. runs `doctor-gpu`, then the no-update `smoke-gpu`;
7. writes complete logs under `results/diagnostics/` and opens the action menu.

The NVIDIA driver is never installed automatically. If WSL2, Docker Desktop, or the
driver was just installed/updated, reboot Windows when instructed and run `START_HERE.cmd`
again. The script is idempotent; closing it does not remove `data/` or `results/`.

`START_HERE.cmd` is deliberately ASCII-only, uses the explicit Windows PowerShell 5.1
path, and contains no locale-dependent CMD text, so GitHub ZIP downloads do not depend on
the machine's legacy code page. If an older ZIP reports `'shell.exe' is not recognized`,
download the current `main` ZIP or bypass only the CMD wrapper with:

```text
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File ".\scripts\setup_friend_pc.ps1" -Action Guided
```

If an older copy says that `nvidia-smi` failed after it already printed a valid RTX 5070,
driver version, and VRAM value, replace the old extracted directory with the current
`main` ZIP. The current script captures the native exit code before processing its output.

Likewise, `Image ... Building` is normal Docker Compose progress, not an error. Older
Windows PowerShell 5.1 wrappers could stop on that stderr line. The current wrapper lets
the native command finish, logs both streams, and decides success from Docker's exit code.
The first pinned PyTorch image download is roughly 4 GB and can take several minutes.
The build checks `sm_120` through PyTorch's compile-time flags because Docker builds do
not expose the host GPU. The subsequent `doctor-gpu` validates the public architecture
list, the visible RTX device, compute capability, VRAM, and mixed-precision support.
The sparse recurrent cell keeps its accumulation state in FP32 and explicitly converts
autocast projection output before indexed injection, so BF16/FP16 smoke and pilot paths
do not fail with an `index_copy_` dtype mismatch.

The menu keeps these operations separate: PC check, image build, GPU doctor, GPU smoke,
UCI info, confirmed UCI download, UCI preparation, pilot dry-run, and confirmed pilot
training. UCI download requires `DOWNLOAD UCI`; training requires `TRAIN PILOT`.

To reopen only the menu:

```text
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\setup_friend_pc.ps1 -Action Menu
```

## Equivalent manual commands

Build and check the pinned runtime:

```text
docker compose --profile research build research-gpu
New-Item -ItemType Directory -Force results\diagnostics
docker compose --profile research run --rm research-gpu doctor-gpu | Tee-Object results\diagnostics\doctor-gpu.log
docker compose --profile research run --rm research-gpu smoke-gpu | Tee-Object results\diagnostics\smoke-gpu.log
```

Stop if `blackwell_ready` is false, CUDA is absent, the GPU is not the expected RTX 5070/
50-series model, or available VRAM is materially below that model's specification.
`smoke-gpu` performs one tiny mixed-
precision forward/backward and reports `optimizer_step: false`; it is not training.

## UCI technical pilot data

Review metadata first; this is network-free:

```text
docker compose --profile research run --rm research-gpu data download uci_energy_efficiency
```

Only after reviewing the official URL, CC BY 4.0 license, disk space, and checksum policy:

```text
docker compose --profile research run --rm research-gpu data download uci_energy_efficiency --confirm-download
docker compose --profile research run --rm research-gpu data prepare-uci-energy
```

The download manifest pins the received bytes. Preparation creates
`uci_energy_efficiency.jsonl.manifest.json` and `.rejected.jsonl` with source/version/
license, raw and prepared SHA256, accepted/rejected counts, and reasons. UCI's API does
not publish an advance digest, so no digest is invented.
The pilot refuses to start if this preparation manifest is missing, incomplete, or does
not match the configured version and prepared-file SHA256.

UCI is an end-to-end technical regression pilot, not a mathematical benchmark. Its
`held_out_geometry` split excludes complete building-geometry groups, preserves original
feature-only prompts, and never injects Y1. Its metric is named held-out geometry OOD,
not mathematical or compositional OOD.

## Dry-run, then separately confirmed pilot

`pilot_gpu` uses at most 768 UCI rows and a 128-node synthetic graph, matching the
generator's hard bound. First inspect configuration and the resource estimate:

```text
docker compose --profile research run --rm research-gpu --profile pilot_gpu show-config
docker compose --profile research run --rm research-gpu --profile pilot_gpu train
```

The second command must print `DRY-RUN COMPLETE`; it must not create a run. Only after
reviewing nodes, edges, batch, sequence length, RAM, and VRAM may a human start training:

```text
docker compose --profile research run --rm research-gpu --profile pilot_gpu train --confirm-train --run-id pilot_gpu_seed1701
```

Never reuse a run ID. Do not launch `paper_gpu` from this guide. That profile additionally
requires `--confirm-heavy-run --accept-resource-estimate`, CUDA, and a prepared local
FlyWire NPZ.

## Next mathematical dataset (offline preparation only)

DeepMind Mathematics v1.0 is pinned to official source commit
`427f45075f84b8b9774950196ad63867ca20ffb3`. The official README links a GCS browser but
does not provide a single pinned direct archive/checksum, so this project will not
download it automatically. After independently obtaining an official local archive:

```text
docker compose --profile research run --rm research-gpu data prepare-deepmind-numeric --source data/raw/deepmind/official-v1.tar.gz
```

This keeps only directly parseable finite numeric answers and writes each official train,
interpolation, and extrapolation split separately. It is not connected to pilot training.

## Optional authorized local FlyWire export

The project does not access FlyWire. If the owner already has an authorized FAFB v783
aggregate with `pre_group,post_group,synapse_count` columns:

```text
docker compose --profile research run --rm research-gpu graph convert-flywire --edges data/raw/flywire/authorized_v783_groups.csv --output data/processed/flywire_fafb_v783.npz
```

The NPZ, node map, and provenance manifest remain local. Do not redistribute them unless
the source terms permit it.

## Return package after the pilot

Send the repository owner:

- all files under `results/diagnostics/`, including `friend-setup.log` and the pilot log;
- the commit from `git rev-parse HEAD` (or the ZIP filename/version);
- `data/raw/uci_energy_efficiency/data.csv.download.json`;
- `data/processed/uci_energy_efficiency.jsonl.manifest.json`;
- the used manifest from `data/processed/splits/`;
- from `results/<pilot_run_id>/`: `config.resolved.yaml`, `run_manifest.json`,
  `summary.json`, `metrics.json`, `history.csv`, and `predictions.jsonl`;
- complete error logs instead of partial conclusions if anything stops.

Send `checkpoint.pt` only when continuation is needed. Do not send licensed FlyWire raw
exports or prepared connectomes without redistribution permission.
