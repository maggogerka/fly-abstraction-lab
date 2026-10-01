# Research protocol

## Research question

Does a neural network whose recurrent topology is constrained by a Drosophila
connectome show stronger systematic generalization on symbolic physics and mathematics
tasks than matched random graphs and conventional neural networks?

This project tests operational behavior only. It does not claim that a fly, its brain,
or a connectome-constrained model “understands” mathematics.

## Confirmatory hypotheses

- **H1:** A trainable connectome RNN exceeds a degree/size-matched random-graph RNN on
  the primary metric under matched data, optimization, parameter budget, and seeds.
- **H2:** A fixed connectome reservoir exceeds a matched random-graph reservoir on the
  primary metric under the same readout and reservoir-scale selection procedure.
- **H3:** Any connectome advantage remains detectable across symbol renaming, numerical
  extrapolation, equivalent forms, rearranged equations, distractors, unseen templates,
  and compositions of known rules.

The null for each comparison is no paired performance advantage across seeds/templates.

## Primary outcome

**Exact solution rate on compositional held-out-template OOD split.**

Exact match is computed from canonical answers. SymPy-equivalent accuracy is a named
secondary metric, never folded into a custom “abstraction score.”

## Models and controls

- FixedConnectomeReservoir
- TrainableConnectomeRNN
- RandomGraphReservoir
- GRU baseline
- MLP baseline
- Graph controls: degree-preserving rewiring, matched Erdos-Renyi, shuffled weights,
  and shuffled directions

Where feasible, comparisons match node count, edge count, hidden width, readout,
training examples, optimizer schedule, early-stopping rule, and random seeds. Parameter
counts are always reported; unavoidable mismatches are disclosed rather than hidden.

## Splits and leakage control

Every example carries immutable `source_id` and `template_id`. A template assigned to a
held-out split cannot appear in training, including transformed variants. Composite OOD
examples use only transformation rules seen during training while withholding their
specific template composition. Split manifests and content hashes are recorded.

## Statistical analysis

- Aggregate first by `template_id`, then across templates/seeds.
- Report bootstrap confidence intervals with the resampling unit stated.
- Use paired permutation tests for paired model/control outcomes.
- Apply Holm correction within each declared family of comparisons.
- Report sample-efficiency curves as tidy records keyed by model, seed, split, and
  number of training examples.

Analyses distinguish confirmatory from exploratory results. Effect sizes, intervals,
seed-level values, failures, and negative findings are retained. No causal or biological
claim follows from a topology-only performance difference.

## Reproducibility contract

Runs record resolved configuration, Git state, environment/hardware, seeds, dependency
versions, and hashes for data, graph, and configuration. Results are append-only by run
ID. The laptop `local_cpu` profile is a smoke-scale path; claims require preregistered,
multi-seed `paper_gpu` runs performed manually on suitable hardware.

