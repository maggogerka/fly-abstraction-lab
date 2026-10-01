# Repository operating rules

These rules apply to the entire repository.

## Safety boundary

- Never start training, fitting, experiments, sweeps, or the `paper_gpu` profile without a separate explicit user request.
- Never download real datasets, SciBench assets, or FlyWire/connectome data without a separate explicit user request.
- Never run `docker compose up`, build heavy images, wait for training, or automatically resume it.
- A `train` invocation without `--confirm-train` must remain a side-effect-free dry-run.
- Real downloads require `--confirm-download`; `paper_gpu` additionally requires `--confirm-heavy-run` and CUDA.
- Do not overwrite completed result directories.

## Allowed validation

Ruff, pytest, configuration validation, one forward/backward unit-test pass without
`optimizer.step`, CLI `doctor`, `show-config`, `data list`, `data prepare-tiny`, a
guaranteed train dry-run, `docker compose config`, and read-only Git inspection are
allowed. Commits and pushes are allowed. Do not build images or start services.

## Engineering rules

- Target Python 3.11 and keep all paths repository-relative.
- Preserve deterministic seeds, source/template provenance, and train/test isolation.
- Keep CPU tests small enough for an 8 GB Windows laptop.
- Update `PROJECT_STATE.md` after each material implementation stage.
- Never remove an unfamiliar pre-existing file.

