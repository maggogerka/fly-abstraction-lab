# Project state

Last updated: 2026-10-05

## Current branch and scope

- Working branch: `feat/research-pc-readiness`, created from `feat/research-mvp`.
- Open PR: #1, `feat/research-pc-readiness` into `main`; merge is intentionally not
  performed by this work.
- Scientific question: whether connectome-constrained topology affects transfer of
  physics/mathematics abstractions relative to matched controls—not whether a fly brain
  literally understands mathematics.
- Implemented task: finite numeric regression only. Symbolic targets are a future stage.
- No real-data download, Docker image build/pull, GPU command, pilot/paper training, or
  experimental-result analysis was run. Pytest ran only the requested tiny CPU integration
  epochs on an 8-node synthetic graph.

## Research-PC readiness implemented

- CPU/GPU runtimes remain pinned by locks and image digests. The GPU image is official
  PyTorch 2.7.1, CUDA 12.8, cuDNN 9 and asserts compiled `sm_120` support.
- `pilot_gpu` now uses the generator-safe maximum of 128 nodes (expected 256 edges), with
  an integration test that enters `prepare_experiment` and constructs the graph/loaders.
- `doctor-gpu` reports Python, PyTorch, CUDA, GPU, compute capability, VRAM, mixed-
  precision support and Blackwell readiness. `smoke-gpu` performs one tiny forward/
  backward without optimizer creation or parameter update.
- `START_HERE.cmd` and `scripts/setup_friend_pc.ps1` provide a clean-Windows path. They
  check Windows x64, 25 GB disk headroom, `nvidia-smi`/driver/GPU/VRAM, WSL2, Git, Docker
  Desktop, Compose, and the engine; optionally offer Git/Docker through exact-confirmation
  `winget`; never install an NVIDIA driver; build the pinned image; run doctor/smoke; and
  retain logs under `results/diagnostics`.
- Host Python, PyTorch, CUDA Toolkit, and cuDNN are not required. Dataset and training menu
  actions remain separate. UCI needs `DOWNLOAD UCI`; pilot training needs `TRAIN PILOT`.
- Windows ZIP launcher hotfix: `START_HERE.cmd` is ASCII-only, uses the absolute system
  PowerShell 5.1 path, and avoids locale-dependent CMD parsing. The documented direct
  PowerShell command bypasses the wrapper if an old ZIP is still in use.

## Leakage, data, and reproducibility

- UCI no longer stores Y1 in `expression`. Its deterministic `held_out_geometry` strategy
  excludes complete geometry groups, copies test prompts unchanged, records all IDs/rules/
  hashes in the split manifest, and reports a held-out building-geometry OOD metric. It is
  explicitly not called mathematical or compositional OOD.
- Tiny, UCI, and local FlyWire preparation now record source URL/description, version,
  license/terms, SHA256, accepted/rejected counts, and rejection reasons. UCI additionally
  writes a rejected-record JSONL. FlyWire remains local-only, streams aggregation through
  SQLite, writes NPZ plus node mapping/provenance, and performs no authentication/network
  access.
- Added offline preparation of a strict finite-numeric subset of DeepMind Mathematics
  v1.0, pinned to official source commit
  `427f45075f84b8b9774950196ad63867ca20ffb3`. It accepts a local official directory,
  tar/tar.gz/tgz, or ZIP; rejects unsafe archive paths; retains only plain finite numeric
  answers; keeps `train-easy`, `train-medium`, `train-hard`, `interpolate`, and
  `extrapolate` separate; and writes module/filter/rejected/source hashes. It is not part
  of pilot training and has no invented automatic download URL.
- Large connectomes remain compressed NPZ with streaming tensor/file hashing; graph
  controls retain identical input/output mappings.

## Training and checkpoint correctness

- Resource estimates and guards still run before data/graph loading and again on actual
  graph size, covering nodes, edges, batch, sequence length, approximate RAM, and VRAM.
- Best checkpoints now snapshot model, optimizer, scaler, epoch, and validation loss
  together. Resume restores that aligned snapshot; a checkpoint already at/above the
  configured final epoch is rejected with an actionable message.
- A full small CPU integration test checks `summary.json`, `metrics.json`, `history.csv`,
  `predictions.jsonl`, `run_manifest.json`, and checkpoint/resume epoch/optimizer-step
  alignment.

## Validation performed

- `ruff format --check .`: passed (49 files already formatted).
- `ruff check .`: passed.
- `pytest -q`: 52 passed, 1 skipped. The only skip is the Blackwell GPU smoke test because
  this machine has CPU-only PyTorch/CUDA unavailable.
- `doctor`: passed on Python 3.11.9. The existing developer venv has PyTorch 2.14.1+cpu;
  target Docker environments remain pinned to PyTorch 2.7.1.
- `smoke_cpu` show-config and train dry-run: passed and created no run.
- `pilot_gpu` show-config and train dry-run: passed; reported 128 nodes and
  `held_out_geometry`; created no run.
- DeepMind download command dry-run: passed and made no network data request.
- `docker compose config --quiet`: passed.
- Windows PowerShell 5.1 parser and required UTF-8 BOM check: passed. The setup script was
  not executed because it would perform Docker/GPU host actions.
- `git diff --check`: passed.
- `docker buildx build --check` for both Dockerfiles was attempted but Docker Desktop's
  Linux engine is not running, so the client could not connect. No daemon was started;
  the same checks remain mandatory in CI.

## Known limitations

- Actual RTX 5070/50-series behavior, Windows guided setup, image build, `doctor-gpu`, `smoke-gpu`,
  UCI acquisition, and the pilot must be run manually on the friend's PC.
- The UCI API publishes no advance digest. Confirmed acquisition computes and records the
  received SHA256; compare the manifest between machines.
- UCI validates the technical pipeline only. A real mathematical-transfer result needs
  the separately prepared mathematical dataset and an approved experiment plan.
- The official DeepMind README exposes a GCS browser but no single version-pinned direct
  archive/checksum. Preparation is therefore local-only and intentionally rejects
  fractions/symbolic numeric forms such as `1/3`; expanding parsing needs a separate
  reviewed protocol.
- The pilot topology is synthetic. Scientific graph comparisons require an authorized
  local FlyWire export and declared matched variants/seeds.
- Symbolic target training/evaluation remains unimplemented.

## Next manual action

On the RTX 5070/50-series PC, use `FRIEND_QUICKSTART_RU.md` or `RUN_RESEARCH_PC.md`: clone/unpack,
run `START_HERE.cmd`, inspect diagnostics, separately confirm UCI download/preparation,
run the pilot dry-run, then explicitly confirm the pilot only if all checks pass.
