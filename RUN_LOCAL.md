# Local CPU workflow

These instructions target Windows, Python 3.11, 8 GB RAM, and no discrete GPU. Training
is never started by setup or validation commands.

## Option A: Python virtual environment

From the repository root:

```text
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m fly_abstraction doctor
.venv\Scripts\python -m fly_abstraction --profile local_cpu show-config
.venv\Scripts\python -m fly_abstraction data prepare-tiny
.venv\Scripts\python -m fly_abstraction --profile local_cpu train
```

The last command is the required dry-run. After reviewing its parameters, RAM, free
disk, and size estimate, a human can start the bounded run exactly once with:

```text
.venv\Scripts\python -m fly_abstraction --profile local_cpu train --confirm-train --run-id local_cpu_001
```

Choose a new run ID each time. Existing result directories are never overwritten.

## Option B: Docker CPU

Validate Compose first:

```text
docker compose config
```

The human-operated build and checks are:

```text
docker compose build local-cpu
docker compose run --rm local-cpu doctor
docker compose run --rm local-cpu data prepare-tiny
docker compose run --rm local-cpu --profile local_cpu train
```

After reviewing the dry-run, the explicit training command is:

```text
docker compose run --rm local-cpu --profile local_cpu train --confirm-train --run-id local_cpu_001
```

Do not use `docker compose up`: services are designed for one-shot commands and mounted
`data/`/`results/` directories.

