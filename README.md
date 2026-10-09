# Fly Abstraction Lab

Fly Abstraction Lab tests whether **connectome-constrained topology changes transfer of
physics/mathematics abstractions** relative to matched graph controls and conventional
neural baselines. It does not test whether a fly brain literally “understands math,” and
a model difference would not establish animal cognition or a biological causal mechanism.

## Current MVP boundary

The implemented task mode is finite numeric regression (`task.mode: numeric`). Every
target is parsed and validated before encoding; strings that are not numbers and
`NaN`/`Inf` are rejected. Reported metrics are MAE, RMSE, mean relative error, and
tolerance accuracy. Tiny synthetic tasks retain a compositional held-out-template smoke
split. UCI uses a separate `held_out_geometry` split: complete building-geometry groups
are withheld, test prompts stay byte-for-byte unchanged, and Y1 is never copied into an
input or expression field. This is a technical geometry OOD check, not mathematical or
compositional OOD. Symbolic scoring remains a future stage.

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

The GPU image build asserts PyTorch 2.7.1, CUDA 12.8, and compiled `sm_120` support
from PyTorch's compile-time flags, which remain available while Docker builds without a GPU.
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

On a clean 64-bit Windows PC, clone or unpack the repository and double-click
[`START_HERE.cmd`](START_HERE.cmd). It checks disk, the NVIDIA driver, WSL2, Git, Docker
Desktop, Compose, and the Docker engine; then it builds the digest-pinned image and runs
`doctor-gpu` plus `smoke-gpu`. Logs go to `results/diagnostics/`. Git or Docker Desktop
can be offered through `winget` only after an exact typed confirmation. The NVIDIA driver
is never installed automatically.

No host Python, PyTorch, CUDA Toolkit, or cuDNN installation is required. Those components
stay inside the pinned Docker image. Dataset download and training are separate menu
actions and never run during guided setup. See [FRIEND_QUICKSTART_RU.md](FRIEND_QUICKSTART_RU.md)
for the short handoff and [RUN_RESEARCH_PC.md](RUN_RESEARCH_PC.md) for exact manual commands.
`smoke-gpu` performs exactly one tiny forward/backward under autocast, with no optimizer
or parameter update.

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

Preparation writes `.manifest.json` and `.rejected.jsonl` beside the JSONL, including
source URL, version, license, raw/prepared SHA256, accepted/rejected counts, and rejection
reasons. The pilot refuses a missing manifest, a version/hash mismatch, or anything other
than the complete validated 768-row source. UCI is only an end-to-end systems pilot; it
is not the mathematical benchmark.
No raw or prepared real data is committed.

The next mathematical-data stage is a finite-numeric subset of DeepMind Mathematics v1.0,
pinned to official source commit `427f45075f84b8b9774950196ad63867ca20ffb3` (Apache-2.0).
The official README links a GCS browser but does not publish one pinned direct archive URL
and checksum, so this project deliberately has no automatic DeepMind download. After the
user independently obtains an official local archive/directory, preparation is offline:

```text
python -m fly_abstraction data prepare-deepmind-numeric --source data/raw/deepmind/official-v1.tar.gz
```

Only answers that parse directly to finite numbers are retained. `train-easy`,
`train-medium`, `train-hard`, `interpolate`, and `extrapolate` stay in separate files; the
manifest records modules, filter statistics, rejected-record SHA256, and source SHA256.
DeepMind is not wired into automatic pilot training.

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

The scientific FlyWire path uses the public static
[FlyWire Whole-brain Connectome Connectivity Data v783](https://zenodo.org/records/10676866)
record (DOI `10.5281/zenodo.10676866`) and only the 852,022,274-byte
`proofread_connections_783.feather` file. Zenodo publishes MD5
`f48f972d262323a102aed49af1396b8a`; the downloader verifies it and also writes a
streamed SHA256. The record does not declare a license in its metadata, so the project
does not invent one: review the record and Zenodo terms before confirming.

Metadata and resource review are network-free:

```text
python -m fly_abstraction data download flywire_fafb_v783
```

Downloading is a separate human action requiring `--confirm-download`. Preparation is
also a dry-run unless `--confirm-prepare` is present:

```text
python -m fly_abstraction graph prepare-flywire-v783 --source data/raw/flywire_fafb_v783/proofread_connections_783.feather --output data/processed/flywire_v783_core_1024.npz --nodes 1024 --min-pair-synapses 5 --input-node-count 64 --output-node-count 64 --seed 1701
python -m fly_abstraction graph inspect --graph data/processed/flywire_v783_core_1024.npz
```

The preparer reads only the four required Feather columns with PyArrow, rejects invalid
IDs/non-finite or non-positive weights, aggregates each directed pair across neuropils,
then thresholds. `weighted_connected_core_v1` starts from maximum total synaptic
strength, grows through the strongest undirected frontier with numeric root-ID ties, and
records component restarts. The saved graph keeps all directed induced edges, original
weights, and self-loops. Input nodes are ranked by outgoing strength and output nodes by
incoming strength; these are artificial task-independent interfaces, not sensory/motor
annotations. NPZ, node CSV, manifest, stats, source/artifact hashes, selection parameters,
and reachability are written without overwrite.

Start with 512 nodes on an RTX 5070 with about 12 GiB VRAM, then 1024. Consider 2048 only
after measured resource logs; do not attempt 4096 until smaller stages have been reviewed.
`flywire_smoke_gpu` targets the already prepared 512-node graph and remains a train
dry-run by default.

The older offline converter remains available for a user-authorized local FAFB v783 CSV
already aggregated to `pre_group,post_group,synapse_count`:

```text
python -m fly_abstraction graph convert-flywire --edges data/raw/flywire/authorized_v783_groups.csv --output data/processed/flywire_fafb_v783.npz
```

That converter streams the CSV through an on-disk SQLite aggregation, writes NPZ plus a
CSV node map and sidecar provenance manifest, records SHA256 and accepted/rejected row
counts, and refuses overwrite. It does not bypass FlyWire authentication or data-use
terms.

## Models and artifacts

Implemented model families are fixed/trainable connectome reservoirs, a matched random
graph reservoir, GRU, and MLP. The numeric answer head is the only trained/scored MVP
head; the legacy expression head is retained only as an untrained future extension
point.

Confirmed runs are append-only under `results/<run_id>/` and include resolved config,
Git/environment/hardware manifest, hashes for data/config/graph/split, history,
predictions, metrics, summary, and an optional checkpoint.
Best checkpoints are epoch-consistent snapshots: model, optimizer, scaler, epoch, and
best validation loss are captured together and restored together on resume.

## Development checks

```text
ruff format --check .
ruff check .
pytest -q
python -m fly_abstraction doctor
python -m fly_abstraction --profile smoke_cpu show-config
python -m fly_abstraction --profile smoke_cpu train
python -m fly_abstraction --profile pilot_gpu train
python -m fly_abstraction data download deepmind_mathematics
python -m fly_abstraction data download flywire_fafb_v783
python -m fly_abstraction --profile flywire_smoke_gpu train
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
