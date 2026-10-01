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

## Decisions

- Python 3.11 with a `src/` package layout.
- YAML profiles are the only intended difference between laptop and research-PC runs.
- Primary metric: exact solution rate on the compositional held-out-template OOD split.
- Tiny generated data and a synthetic connectome are the only locally prepared assets.

## Validation

- Source and configuration validation is pending runtime discovery.
- No training or download has been run.

## Constraints observed

- No training, real downloads, Docker builds, or services have been started.
- GitHub CLI is installed, but the configured `maggogerka` token is invalid.
- A Python 3.11 runtime was not detected by the Windows Python launcher.

## Next step

Run Ruff, pytest, safe CLI commands, and `docker compose config`; repair any failures;
then commit the MVP and push if GitHub authentication becomes valid.

