# Fly Abstraction Lab

Fly Abstraction Lab tests whether **connectome-constrained topology changes transfer of
physics/mathematics abstractions** relative to matched graph controls and conventional
neural baselines. It does not test whether a fly brain literally “understands math,” and
a model difference would not establish animal cognition or a biological causal mechanism.

## Current MVP boundary

The implemented task mode is finite numeric regression (`task.mode: numeric`). Every
target is parsed and validated before encoding; strings that are not numbers and
`NaN`/`Inf` are rejected. Reported metrics are MAE, RMSE, mean relative error, and
tolerance accuracy. The primary pilot outcome is tolerance accuracy on the compositional
held-out-template OOD split. Symbolic target generation/scoring is a separately marked
future stage and is not represented as working support.

Every split is stored as a deterministic manifest containing example IDs, dataset
version, seed, explicit rules, and SHA256. Repeating the same inputs and rules reuses an
identical manifest.

## Environments

- CPU: Python 3.11.9 and PyTorch 2.7.1 CPU from `requirements-cpu.lock`.
- GPU: official
  [`pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`](https://hub.docker.com/r/pytorch/pytorch/tags)
  image pinned to digest
  `sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2`.
- PyTorch 2.7 introduced Blackwell support and CUDA 12.8 wheels; see the
  [official PyTorch 2.7 release post](https://pytorch.org/blog/pytorch-2-7/).

The GPU image build asserts PyTorch 2.7.1, CUDA 12.8, and compiled `sm_120` support.
`doctor-gpu` additionally checks the visible GPU, VRAM, compute capability, FP16/BF16/
TF32 availability, and Blackwell readiness at runtime.

## Laptop: safe quick start

Windows PowerShell, Python 3.11:

```text
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-cpu.lock
.venv\Scripts\python -m pip install --no-build-isolation --no-deps -e .
.venv\Scripts\python -m fly_abstraction doctor
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m fly_abstraction --profile smoke_cpu train
```

The last command is a dry-run: it prints the resolved parameters and resource estimate,
does not enter the training function, and creates no run directory. See
[RUN_LOCAL.md](RUN_LOCAL.md) for the optional explicitly confirmed tiny CPU run.

## RTX research PC

Use [RUN_RESEARCH_PC.md](RUN_RESEARCH_PC.md). The intended order is image build,
`doctor-gpu`, `smoke-gpu`, data preparation, pilot dry-run, then a separately confirmed
pilot. `smoke-gpu` performs exactly one tiny forward/backward under autocast, with no
optimizer and no parameter update.

`paper_gpu` is deliberately blocked until all of the following are present:

- CUDA is visible;
- the estimator reports nodes, edges, batch, sequence length, RAM, and VRAM within the
  configured headroom;
- `--confirm-train`, `--confirm-heavy-run`, and `--accept-resource-estimate` are all
  supplied.

## Data

The real numeric pilot dataset is
[UCI Energy Efficiency (ID 242)](https://archive.ics.uci.edu/dataset/242/energy+efficiency),
version `UCI-242-2024-02-26`, DOI `10.24432/C51307`, CC BY 4.0. It has 768 rows, eight
numeric features, and two numeric targets; the MVP predicts heating load (`Y1`).

Importing the package never downloads it. First review metadata/disk use:

```text
python -m fly_abstraction data download uci_energy_efficiency
```

Only a human should then add `--confirm-download`. The command accepts only the pinned
official UCI URL, streams to a temporary file, refuses overwrite, computes SHA256, and
writes an adjacent download manifest with source, version, license, byte size, and hash.
UCI's API does not publish a pre-download digest, so this limitation is explicit; the
first confirmed download pins the received bytes rather than inventing a checksum.

Convert the confirmed CSV to validated numeric JSONL with:

```text
python -m fly_abstraction data prepare-uci-energy
```

No raw or prepared real data is committed.

## Graphs and controls

Large graphs use compressed NumPy NPZ, not JSON. Hashing streams contiguous tensor bytes
without converting edge tensors to Python lists. Supported `graph.variant` values are:

- `real`
- `degree_preserving`
- `weight_shuffled`
- `direction_shuffled`
- `er_random`

Every control retains the exact input-node and output-node mappings. The two random
topology controls are explicitly directed multigraph controls, which makes generation
linear in edge count and avoids large Python edge sets.

FlyWire access and authorization are intentionally outside this repository. Given a
user-authorized local FAFB v783 CSV already aggregated to columns
`pre_group,post_group,synapse_count`, convert it without any network access:

```text
python -m fly_abstraction graph convert-flywire --edges data/raw/flywire/authorized_v783_groups.csv --output data/processed/flywire_fafb_v783.npz
```

The converter streams the CSV through an on-disk SQLite aggregation, writes NPZ plus a
CSV node map, records the source SHA256, and refuses overwrite. It does not bypass
FlyWire authentication or data-use terms.

## Models and artifacts

Implemented model families are fixed/trainable connectome reservoirs, a matched random
graph reservoir, GRU, and MLP. The numeric answer head is the only trained/scored MVP
head; the legacy expression head is retained only as an untrained future extension
point.

Confirmed runs are append-only under `results/<run_id>/` and include resolved config,
Git/environment/hardware manifest, hashes for data/config/graph/split, history,
predictions, metrics, summary, and an optional checkpoint.

## Development checks

```text
ruff format --check .
ruff check .
pytest -q
python -m fly_abstraction doctor
python -m fly_abstraction --profile smoke_cpu show-config
python -m fly_abstraction --profile smoke_cpu train
docker compose config --quiet
docker buildx build --check --file Dockerfile.cpu .
docker buildx build --check --file Dockerfile.gpu .
```

CI runs lint, unit tests (GPU tests skip when CUDA is absent), CLI dry-runs, and Dockerfile
build checks. It performs no training, real-data downloads, GPU work, or image-layer
builds.

The detailed hypothesis and analysis boundary is in
[RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md). Project code is MIT; external datasets and
connectome exports retain their own terms.
