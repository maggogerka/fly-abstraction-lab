# RTX 5090 research-PC workflow

These are manual commands for a Windows/PowerShell host with 32 GB RAM, Docker Desktop,
the NVIDIA Container Toolkit integration, and a driver new enough to run CUDA 12.8
containers. They are not run by setup, tests, or CI.

## 1. Clone the exact branch

```text
git clone https://github.com/maggogerka/fly-abstraction-lab.git
cd fly-abstraction-lab
git switch feat/research-pc-readiness
git rev-parse HEAD
git status --short
```

Keep the printed commit hash with the results. The worktree should be clean.

## 2. Build and inspect the pinned GPU environment

The Dockerfile uses the official PyTorch 2.7.1/CUDA 12.8/cuDNN 9 image pinned by digest
and verifies that `sm_120` is compiled in.

```text
docker compose --profile research build research-gpu
New-Item -ItemType Directory -Force results\diagnostics
docker compose --profile research run --rm research-gpu doctor-gpu | Tee-Object results\diagnostics\doctor-gpu.json
```

Stop if `blackwell_ready` is false, CUDA is absent, the GPU name is unexpected, or VRAM
is materially below the RTX 5090's expected capacity. Do not paper over a failed doctor
with a different unpinned PyTorch install.

## 3. Run the no-training GPU smoke check

```text
docker compose --profile research run --rm research-gpu smoke-gpu | Tee-Object results\diagnostics\smoke-gpu.json
```

Expected fields are `forward: ok`, `backward: ok`, finite loss/gradient, and
`optimizer_step: false`. This is one tiny forward/backward only.

## 4. Prepare the real numeric pilot data

Review official UCI metadata first; this makes no network data request:

```text
docker compose --profile research run --rm research-gpu data download uci_energy_efficiency
```

After independently reviewing the URL, CC BY 4.0 license, expected size, and free disk,
the human-authorized download is:

```text
docker compose --profile research run --rm research-gpu data download uci_energy_efficiency --confirm-download
docker compose --profile research run --rm research-gpu data prepare-uci-energy
```

Retain `data/raw/uci_energy_efficiency/data.csv.download.json`. It records the SHA256 of
the received official bytes. The UCI API does not publish an advance checksum; this is
documented rather than replaced with an invented value.

## 5. Review the small pilot

`pilot_gpu` uses 768 numeric UCI examples and a 256-node synthetic graph. It validates
the GPU/numeric/split/artifact path; by itself it is not evidence for a biological or
connectome-topology claim.

```text
docker compose --profile research run --rm research-gpu --profile pilot_gpu show-config
docker compose --profile research run --rm research-gpu --profile pilot_gpu train
```

The second command must say `DRY-RUN COMPLETE` and display nodes, edges, batch, sequence
length, estimated RAM/VRAM, and available memory. If it reports a guard issue, reduce the
corresponding values; do not bypass the guard.

## 6. Start the pilot only after review

This is the only command below that trains:

```text
docker compose --profile research run --rm research-gpu --profile pilot_gpu train --confirm-train --run-id pilot_gpu_seed1701
```

Do not reuse a run ID. Do not launch `paper_gpu` from this guide. The paper profile also
requires `--confirm-heavy-run --accept-resource-estimate`, CUDA, and a prepared local
FlyWire NPZ; it should be reviewed separately after the pilot.

## Optional: authorized local FlyWire export

The project does not access FlyWire. If the owner has already exported an authorized
FAFB v783 aggregate with `pre_group,post_group,synapse_count` columns:

```text
docker compose --profile research run --rm research-gpu graph convert-flywire --edges data/raw/flywire/authorized_v783_groups.csv --output data/processed/flywire_fafb_v783.npz
```

The resulting NPZ and `.nodes.csv` remain local and must not be redistributed unless the
source terms permit it.

## Return package after the pilot

Send the repository owner:

- `results/diagnostics/doctor-gpu.json`
- `results/diagnostics/smoke-gpu.json`
- the commit hash from `git rev-parse HEAD`
- `data/raw/uci_energy_efficiency/data.csv.download.json`
- `data/processed/splits/*.json`
- from `results/pilot_gpu_seed1701/`: `config.resolved.yaml`, `run_manifest.json`,
  `summary.json`, `metrics.json`, `history.csv`, and `predictions.jsonl`
- `checkpoint.pt` only if model continuation is required (it is larger and is not
  needed for a first metrics review)
- the complete terminal error text instead of partial artifacts if the run stops

Do not send licensed FlyWire raw exports or prepared connectomes without confirming that
redistribution is allowed.
