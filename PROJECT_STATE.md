# Project state

Last updated: 2026-10-01

## Completed

- Captured repository-wide safety constraints in `AGENTS.md`.
- Defined the scientific hypotheses and analysis contract in `RESEARCH_PROTOCOL.md`.
- Added initial project metadata, licensing, and ignore rules.
- Added Python 3.11 `src/` package metadata and merged YAML profiles for `local_cpu`
  and `paper_gpu`, including enforceable laptop limits and two-step heavy-run gating.
- Implemented canonical `MathProblem`, adapters for DeepMind Mathematics,
  SRSD-Feynman, SciBench, and deterministic TinyDataset generation.
- Implemented deterministic symbol/numeric/equivalence/equation/distractor/template/
  compositional OOD handling, SymPy equivalence checks, and template leakage guards.
- Implemented directed `ConnectomeGraph`, manifest/hash, filtering, normalization,
  aggregation, a <=128-node synthetic graph, and local-only FlyWire FAFB v783 adapter.
- Implemented degree-preserving, Erdos-Renyi, weight-shuffled, and direction-shuffled
  controls with invariant checks.
- Implemented fixed/trainable connectome models, random reservoir, GRU and MLP with
  symbolic plus numeric inputs, answer/expression heads, sparse edge-only recurrence,
  parameter counts, device resolution, and finite-value checks.
- Implemented (but did not run) AdamW training/validation/test, clipping, early stopping,
  resume, deterministic seeds, mixed precision, metrics, and append-only run artifacts.
- Implemented bootstrap intervals, paired permutation test, Holm correction,
  template aggregation, and sample-efficiency record format.
- Added safety-gated CLI, CPU/GPU Docker definitions, Compose, CPU-only CI, tests,
  local/research-PC guides, and citation metadata.
- Created private GitHub repository `maggogerka/fly-abstraction-lab`, attached `origin`,
  and pushed `main` plus `feat/research-mvp` without creating a PR or merge.

## Decisions

- Python 3.11 with a `src/` package layout.
- YAML profiles are the only intended difference between laptop and research-PC runs.
- Primary metric: exact solution rate on the compositional held-out-template OOD split.
- Tiny generated data and a synthetic connectome are the only locally prepared assets.

## Validation

- Installed Python 3.11.9 through the official Python Install Manager and created the
  ignored local `.venv`; project dependencies installed successfully.
- `ruff check .`: passed.
- `pytest`: 16 passed in 11.20s, including exactly one forward/backward without an
  optimizer step.
- `doctor`: passed on Python 3.11.9 / PyTorch 2.14.1+cpu; CUDA correctly unavailable.
- `show-config`: passed for `local_cpu`, with every required limit resolved correctly.
- `data list`: passed; only registry metadata was read.
- `data prepare-tiny`: generated 100 deterministic ignored JSONL records locally.
- `train` without confirmation: passed as a side-effect-free dry-run; no result run
  directory was created.
- `docker compose config --quiet`: passed; no image was built and no service started.
- GitHub remote verified private and both requested branches pushed successfully.
- No training or download has been run.

## Constraints observed

- No training, real downloads, Docker builds, or services have been started.
- The real dataset/connectome adapters require separately authorized, license-compliant
  local artifacts before a research run.

## Next step

A human may run the confirmed `local_cpu` command and analyze the append-only artifacts
under `results/<run_id>/`; paper-scale assets and execution remain explicitly deferred.

