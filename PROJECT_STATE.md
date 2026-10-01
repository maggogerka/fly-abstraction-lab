# Project state

Last updated: 2026-10-01

## Current branch and scope

- Working branch: `feat/research-pc-readiness`, created from `feat/research-mvp`.
- Scientific question: whether connectome-constrained topology affects transfer of
  physics/mathematics abstractions relative to matched controls—not whether a fly brain
  literally understands mathematics.
- Implemented MVP task: finite numeric regression only. Symbolic targets are explicitly
  a future stage.
- No training, real-data download, GPU task, Docker layer build, or result analysis was
  run during this stage.

## Implemented in this stage

- Pinned Python/PyTorch/dependency environments with `requirements.lock` and
  `requirements-cpu.lock`; pinned CPU and GPU base-image digests.
- Updated GPU runtime to official PyTorch 2.7.1, CUDA 12.8, cuDNN 9 with `sm_120`
  assertion for Blackwell/RTX 5090.
- Added `doctor-gpu` and `smoke-gpu`; the latter performs one tiny mixed-precision
  forward/backward without optimizer creation or parameter update.
- Added `smoke_cpu`, `smoke_gpu`, and bounded `pilot_gpu` profiles. Reduced the former
  paper profile to finite numeric data/5k graph nodes and added three independent gates:
  confirmed training, heavy-run confirmation, and resource-estimate acceptance.
- Added conservative RAM/VRAM estimation from graph nodes/edges, batch, sequence length,
  activations, graph storage, and optimizer/model state. Both CLI and training API stop
  unsafe runs before training; actual graph size is checked again after loading.
- Enforced `task.mode: numeric`; invalid numeric strings and `NaN`/`Inf` now fail during
  schema validation instead of entering tensors. Added MAE, RMSE, mean relative error,
  and tolerance accuracy with configured tolerances.
- Added deterministic order-independent split manifests with IDs, dataset/version, seed,
  explicit rules, source-ID SHA256, and manifest SHA256. Existing manifests are reused
  only when identical.
- Added selectable graph variants: `real`, `degree_preserving`, `weight_shuffled`,
  `direction_shuffled`, and `er_random`. Controls preserve edge/node counts, weights, and
  identical input/output mappings; models now use those mappings explicitly.
- Replaced large-graph JSON as the primary format with compressed NPZ. Graph hashing
  streams tensor bytes without `.tolist()`. Random controls use tensor operations instead
  of large Python edge lists/sets.
- Added consent-gated UCI Energy Efficiency acquisition from the official pinned URL.
  It requires `--confirm-download`, streams with a size cap, refuses overwrite, computes
  SHA256, and records source/version/license/hash in an adjacent manifest. UCI does not
  publish an advance digest in its API; this limitation is recorded explicitly.
- Added UCI numeric preparation for 768 rows and a local-only FlyWire FAFB v783 converter
  for authorized aggregated CSV exports. The converter uses on-disk SQLite aggregation,
  NPZ output, a CSV node map, and source SHA256; it performs no network/authentication.
- Expanded tests for numeric/finite targets, metrics, split reproducibility, graph
  variants and I/O mappings, binary graph hashes, resource guard, CLI safety, local
  FlyWire conversion, streaming download checksums, and GPU skip behavior.
- CI now checks formatting, lint, unit tests, CLI doctor/config/train dry-run, Compose
  syntax, and Dockerfile build checks without training or data downloads.
- Updated README, research-PC/local runbooks, protocol, and operating rules.

## Validation performed

- `ruff format --check .`: passed (45 files formatted).
- `ruff check .`: passed.
- `pytest -q`: 40 passed, 1 skipped; the only skip is the Blackwell GPU smoke test because
  this machine has CPU-only PyTorch/CUDA unavailable.
- `doctor`: passed on Python 3.11.9; local PyTorch is 2.14.1+cpu and CUDA is correctly
  unavailable. The target Docker environments remain pinned to PyTorch 2.7.1.
- `show-config` and side-effect-free `train` dry-run passed for `smoke_cpu`; the dry-run
  printed nodes, edges, batch, sequence length, RAM/VRAM estimates and created no run.
- UCI download dry-run passed and performed no download.
- `docker compose config --quiet`: passed.
- Official image manifests were resolved without pulling layers: GPU digest
  `sha256:c16f4c...e2a2`; CPU multi-platform digest `sha256:8fb099...c317`.
- Local `docker buildx build --check` for both Dockerfiles could not connect because
  Docker Desktop's Linux engine is not running. No daemon was started automatically;
  equivalent checks are configured in CI.

## Known limitations

- `smoke-gpu` and actual RTX 5090 behavior must be run on the friend's machine.
- The UCI API exposes version/license/source metadata but no published pre-download
  digest. The confirmed command pins the received bytes by SHA256; compare that manifest
  between machines before using the dataset.
- The pilot uses a small synthetic topology to validate the stack. Scientific graph
  comparisons require an authorized local FlyWire export, declared variants/seeds, and
  a separately approved experiment plan.
- Directed degree-preserving and ER controls are multigraph null models; duplicates and
  self-loops are documented where applicable and should be considered in interpretation.
- Symbolic target training/evaluation is not implemented.

## Next manual action

On the RTX 5090 PC, follow `RUN_RESEARCH_PC.md`: clone the branch, build the pinned image,
run `doctor-gpu`, run `smoke-gpu`, explicitly prepare UCI data, inspect the pilot dry-run,
and only then decide whether to issue the confirmed pilot command.
