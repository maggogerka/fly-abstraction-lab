# Research PC workflow (future GPU runs)

Nothing in this document is executed automatically. Use it later on the RTX 5090 host.

## Transfer the exact code

```text
git clone https://github.com/maggogerka/fly-abstraction-lab.git
cd fly-abstraction-lab
git switch feat/research-mvp
git rev-parse HEAD
```

Install Python 3.11 and a PyTorch build compatible with the host driver/CUDA runtime,
then install the project. Do not change Python code between machines; place reviewed
hardware/experiment values in `configs/paper_gpu.yaml` or a separate YAML override.

## Prepare external assets manually

1. Review registry URLs, citations, licenses, expected sizes, RAM, and free disk.
2. Acquire datasets only after explicit authorization and convert them to canonical
   JSONL `MathProblem` records with stable source/template IDs.
3. Export the permitted FlyWire FAFB v783 graph locally with its release manifest.
4. Hash and retain the raw/prepared manifests. Never commit licensed raw data.
5. Run leakage checks before any experiment.

## Validate without training

```text
python -m fly_abstraction --profile paper_gpu doctor
python -m fly_abstraction --profile paper_gpu show-config
python -m fly_abstraction --profile paper_gpu train
```

The third command is still a dry-run. Only a human who has reviewed the resolved config
and resources should later add both gates:

```text
python -m fly_abstraction --profile paper_gpu train --confirm-train --confirm-heavy-run --run-id paper_gpu_seed1701
```

The command refuses to proceed without CUDA. Use unique run IDs and push code/config
commits before recording confirmatory results. Do not merge exploratory and confirmatory
families when applying Holm correction.

