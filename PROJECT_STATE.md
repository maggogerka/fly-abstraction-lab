# Project state

Last updated: 2026-10-09

## Current branch and safety boundary

- Working branch: `feat/flywire-v783-subgraph`, created from `main`.
- Open PR: [#8](https://github.com/maggogerka/fly-abstraction-lab/pull/8) from
  `feat/flywire-v783-subgraph` into `main`. Merge is not part of this stage.
- No real FlyWire/DeepMind/UCI data was downloaded. No Docker image or service was
  built/started. No CUDA command, confirmed training, pilot, sweep, paper run, or result
  analysis was executed.
- Scientific question: whether connectome-constrained topology affects transfer of
  physics/mathematics abstractions relative to matched controls. This does not test
  literal mathematical understanding by a fly or establish a biological mechanism.
- The MVP remains finite numeric regression only; symbolic targets are future work.

## FlyWire FAFB v783 stage

- Registered the public static Zenodo v783 record, DOI
  `10.5281/zenodo.10676866`, file `proofread_connections_783.feather`, exact size
  852,022,274 bytes, and published MD5 `f48f972d262323a102aed49af1396b8a`.
  The record does not state a data license, so the registry records that limitation and
  links the source terms rather than inventing a license.
- Confirmed download streams to a partial file, verifies the published MD5, always
  computes SHA256, writes an adjacent provenance manifest, and refuses overwrite.
- Pinned `pyarrow==20.0.0` in project/runtime/CPU dependency declarations. Preparation
  scans only pre/post root ID, neuropil, and synapse-count columns; validates types and
  values; aggregates pairs across neuropils with compact NumPy arrays; thresholds after
  aggregation; and performs a conservative RAM preflight.
- Implemented `weighted_connected_core_v1` with strength ranking, numeric root-ID ties,
  undirected frontier growth, recorded component restarts, and an exact requested node
  count. The saved graph retains directed induced edges, weights, and self-loops.
- Input nodes rank by outgoing strength; output nodes rank by incoming strength and avoid
  inputs when enough alternatives exist. Mappings are explicitly artificial and
  task-independent. All graph controls preserve them.
- Artifacts are byte-deterministic NPZ plus `.nodes.csv`, `.manifest.json`, and
  `.stats.json`. Provenance includes source/artifact hashes, all parameters, row/pair
  counts, mappings, component/reachability statistics, selected-root hash, software, and
  Git commit. Existing/partial artifacts are not overwritten.
- `graph inspect` is read-only and reports node/edge counts, density, self-loops,
  interface sizes, degree/weight summaries, weak components, directed reachability,
  release/algorithm/source hash, graph content hash, and NPZ SHA256.

## Pipeline and operator workflow

- The FlyWire adapter now loads the prepared NPZ exactly. It rejects a graph larger than
  `graph.max_graph_nodes` and never silently keeps the first N nodes.
- Added `flywire_smoke_gpu`: prepared 512-node graph, tiny deterministic numeric data,
  one epoch, batch 2, CUDA mixed precision, and resource guard. `train` remains a
  side-effect-free dry-run unless separately confirmed.
- The Windows menu separates metadata, confirmed download, 512/1024 preparation
  dry-runs, confirmed preparations, inspections, training dry-run, and confirmed
  one-epoch smoke. Each destructive/heavy boundary uses a distinct exact phrase.
- Recommended progression for the RTX 5070 (~12 GiB) is 512, then 1024; 2048 requires
  measured headroom, and 4096 is deferred until smaller measurements justify it.

## Test coverage

- Synthetic Feather tests cover schema rejection, invalid-row accounting, aggregation
  across neuropils, threshold order, row-order invariance, deterministic core selection,
  exact size/induced edges/self-loop retention, deterministic NPZ bytes, mappings and
  controls, provenance/hashes, inspect, overwrite refusal, dry-run no-write behavior,
  exact pipeline loading, and one CPU forward/backward without an optimizer step.
- CI includes lint, all unit tests, CLI doctor/config/train dry-runs, both real-data
  metadata dry-runs, Compose validation, and Dockerfile build checks. It performs no
  dataset download, training, GPU work, service start, or image-layer build.

## Validation status

- `ruff format --check .`: passed (51 files).
- `ruff check .`: passed.
- `pytest -ra`: 60 passed, 1 skipped. The skip is the CUDA/Blackwell runtime test because
  this laptop has CPU-only PyTorch and no CUDA device.
- `python -m fly_abstraction doctor`: passed on Python 3.11.9.
- `python -m fly_abstraction --profile smoke_cpu train`: passed as a dry-run and created
  no result directory.
- FlyWire metadata download command, `flywire_smoke_gpu show-config`, and
  `flywire_smoke_gpu train`: passed as metadata/config/dry-run operations; no network
  download or training occurred.
- `docker compose config --quiet`: passed.
- Windows PowerShell parser and UTF-8 BOM check: passed. The setup script was not run.
- `git diff --check`: passed (Git reported only its normal LF-to-CRLF checkout warning).
- Dockerfile `buildx --check` was attempted for CPU and GPU files but Docker Desktop's
  Linux engine is not running. No daemon was started and no image layers were built; CI
  remains responsible for these two checks.

## Known limitations

- The official 852 MB file and real 512/1024 cores have not been downloaded or prepared;
  synthetic fixtures validate behavior, not real-data statistics.
- RTX 5070 Docker/runtime validation and the confirmed one-epoch smoke remain manual.
- Core selection is intentionally strength/connectedness biased and is not representative
  sampling. Scientific claims require matched controls, declared seeds, and later
  approved experiments.
- DeepMind remains offline-only and is not connected to this stage.
