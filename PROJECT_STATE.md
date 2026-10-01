# Project state

Last updated: 2026-10-01

## Completed

- Captured repository-wide safety constraints in `AGENTS.md`.
- Defined the scientific hypotheses and analysis contract in `RESEARCH_PROTOCOL.md`.
- Added initial project metadata, licensing, and ignore rules.

## Decisions

- Python 3.11 with a `src/` package layout.
- YAML profiles are the only intended difference between laptop and research-PC runs.
- Primary metric: exact solution rate on the compositional held-out-template OOD split.
- Tiny generated data and a synthetic connectome are the only locally prepared assets.

## Validation

- Not run yet.

## Constraints observed

- No training, real downloads, Docker builds, or services have been started.
- GitHub CLI is installed, but the configured `maggogerka` token is invalid.
- A Python 3.11 runtime was not detected by the Windows Python launcher.

## Next step

Implement the package, profiles, tests, containers, CI, and user documentation; then
run only the explicitly permitted validation commands.

