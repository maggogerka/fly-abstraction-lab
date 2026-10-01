# Fly Abstraction Lab

Fly Abstraction Lab is a reproducible research platform for testing whether a neural
network whose recurrent topology is constrained by a Drosophila connectome exhibits
stronger systematic physics/mathematics generalization than matched random graphs and
standard neural baselines.

The project tests operational behaviors: transfer to new numbers, renamed symbols,
equivalent expressions, rearranged equations, unseen templates, compositions of known
rules, and irrelevant variables. It does **not** claim that a fly, a connectome, or a
trained network understands mathematics. A topology advantage would be an empirical
model result, not evidence of animal cognition or a biological causal mechanism.

## Confirmatory target

The primary metric is **Exact solution rate on compositional held-out-template OOD
split**. SymPy-equivalent accuracy and transformation-specific results are secondary.
No arbitrary aggregate “abstraction score” is used. The full hypotheses, matching
rules, leakage controls, and interpretation policy are in [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md).

## Data

Every source is converted to the immutable `MathProblem` schema and retains
`source_id`/`template_id` provenance.

- [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset)
  (Apache-2.0 code/data generator; Saxton et al., 2019)
- [SRSD-Feynman](https://github.com/omron-sinicx/srsd-benchmark) (external scientific
  symbolic-regression source; Matsubara et al., 2024)
- [SciBench](https://github.com/mandyyyyii/scibench) (optional external benchmark;
  Wang et al., ICML 2024)
- `TinyDataset`: at most 100 deterministic, locally generated wiring-test problems

The registry records URL, version, license note, citation, adapter, and expected size.
No real dataset or FlyWire artifact is bundled or downloaded automatically. Review the
upstream terms and source-text rights before acquisition or redistribution.

## Graphs and models

`ConnectomeGraph` stores directed `edge_index`, `edge_weight`, metadata, and a manifest.
It supports filtering, incoming/outgoing normalization, aggregation, content hashing,
and a future local FlyWire FAFB v783 export. The included tiny graph never exceeds 128
nodes. Controls are degree-preserving rewiring, matched directed Erdos-Renyi, shuffled
weights, and shuffled directions.

Implemented models share answer and expression-token heads:

- `FixedConnectomeReservoir`
- `TrainableConnectomeRNN`
- `RandomGraphReservoir`
- GRU baseline
- MLP baseline

The graph RNN has one recurrent parameter per existing edge and no trainable off-graph
recurrent matrix. A registry is provided for later LSTM and SmallTransformer baselines.

## Safety-first quick start

Requires Python 3.11.

```text
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m fly_abstraction doctor
.venv\Scripts\python -m fly_abstraction data prepare-tiny
.venv\Scripts\python -m fly_abstraction --profile local_cpu train
```

The final command is a dry-run. It does not construct a result directory or enter the
training function. To manually authorize the bounded laptop run:

```text
.venv\Scripts\python -m fly_abstraction --profile local_cpu train --confirm-train --run-id local_cpu_001
```

Never add confirmation flags in unattended automation. `paper_gpu` additionally needs
`--confirm-heavy-run` and an available CUDA device. Local safety limits cannot be
exceeded without `--override-local-safety`. Result directories are append-only.

See [RUN_LOCAL.md](RUN_LOCAL.md) for venv and Docker CPU procedures and
[RUN_RESEARCH_PC.md](RUN_RESEARCH_PC.md) for the manual RTX research workflow.

## Reproducibility

YAML profiles are resolved from `configs/base.yaml` plus a hardware profile. Runs fix
Python/NumPy/PyTorch seeds and can enable deterministic algorithms. Each confirmed run
writes under `results/<run_id>/`:

- `summary.json`, `metrics.json`, `history.csv`
- `run_manifest.json`, `predictions.jsonl`, `config.resolved.yaml`
- `checkpoint.pt` only when the profile enables checkpoints

The manifest includes Git commit/dirty state, OS, CPU, RAM, GPU, Python, PyTorch, CUDA,
seed, dependency versions, and hashes of the data, graph, and configuration. Statistical
helpers provide template-level aggregation, bootstrap intervals, paired permutation
tests, Holm correction, and tidy sample-efficiency points.

## Development checks

```text
ruff check .
pytest
python -m fly_abstraction --profile local_cpu show-config
python -m fly_abstraction data list
docker compose config
```

CPU-only GitHub Actions runs lint, tests, and the guaranteed train dry-run. It never
downloads data, builds GPU images, or starts training.

## License

Project code is MIT licensed. External datasets and connectome exports retain their own
terms; registry metadata is not a substitute for reviewing upstream licenses.

