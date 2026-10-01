# Research protocol

## Research question

Does connectome-constrained recurrent topology affect transfer of physics/mathematics
abstractions relative to matched control topologies and conventional neural networks?

This is an operational model comparison. It does not test whether a fly, its brain, or a
connectome “understands mathematics,” and topology-only performance cannot establish a
biological causal mechanism.

## Staged scope

The first MVP is **numeric regression only**. Targets must parse to finite floats. The
implemented output measures are MAE, RMSE, mean relative error, and tolerance accuracy
with declared absolute/relative tolerances. Symbolic target prediction and symbolic
equivalence scoring are a future stage; they must not be mixed into MVP results.

The UCI Energy Efficiency pilot validates the end-to-end numeric research path. Its
small synthetic graph is a systems pilot, not a confirmatory connectome experiment.

## Confirmatory hypotheses for a later approved study

- **H1:** A trainable connectome RNN exceeds a degree/size-matched topology control on
  numeric transfer under matched data, optimizer, parameter budget, and seeds.
- **H2:** A fixed connectome reservoir exceeds a matched random-graph reservoir under
  the same readout and reservoir-scale selection procedure.
- **H3:** Any topology advantage remains detectable across numerical extrapolation,
  renamed variables, equivalent numeric forms, equation rearrangements, distractors,
  unseen templates, and compositions of rules seen during training.

The null for each comparison is no paired performance advantage across declared
seeds/templates.

## Primary outcome

**Tolerance accuracy on the compositional held-out-template OOD split**, with absolute
and relative tolerances frozen in the resolved configuration. MAE, RMSE, mean relative
error, transformation-specific outcomes, and sample efficiency are secondary. No custom
aggregate “abstraction score” is used.

## Models and graph controls

- FixedConnectomeReservoir
- TrainableConnectomeRNN
- RandomGraphReservoir
- GRU baseline
- MLP baseline
- Graph variants: `real`, `degree_preserving`, `weight_shuffled`,
  `direction_shuffled`, and `er_random`

Controls preserve node count, edge count, edge-weight multiset, and the exact input/output
node mappings. Degree-preserving and ER variants use explicitly declared directed
multigraph null models so that construction remains linear in edge count. Parameter
counts and unavoidable mismatches are reported.

## Splits and leakage control

Each example has immutable `source_id` and `template_id`. Whole held-out templates never
occur in training. Manifests contain dataset name/version, all IDs by split, seed, exact
selection rules, source-ID SHA256, and manifest SHA256. Row order cannot change a split,
and an existing manifest is reused only when its content matches exactly.

## Resource and execution policy

Every confirmed run is preceded by an estimate using nodes, edges, batch size, sequence
length, model state, activation state, dataset size, and configured RAM/VRAM headroom.
The run stops before training if the estimate is unsafe or CUDA is missing. `paper_gpu`
also needs explicit acceptance of the printed estimate and the separate heavy-run gate.

## Statistical analysis

- Aggregate first by `template_id`, then across templates/seeds.
- Report bootstrap confidence intervals with the resampling unit stated.
- Use paired permutation tests for paired model/control outcomes.
- Apply Holm correction within each preregistered comparison family.
- Retain seed-level values, failures, negative findings, and sample-efficiency records.

Confirmatory and exploratory results remain separate. A topology effect is reported as a
model result, not as animal cognition or biological explanation.

## Reproducibility contract

Runs record resolved config, Git state, environment/hardware, seeds, pinned dependency
versions, and SHA256 hashes for data, graph, configuration, and split manifest. Graphs use
binary NPZ with streamed byte hashing. Results are append-only by run ID. Raw datasets and
licensed connectome exports are neither committed nor redistributed by this project.
