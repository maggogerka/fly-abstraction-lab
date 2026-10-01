"""Metadata-only registry. Importing it never downloads data."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    name: str
    url: str
    version: str
    license: str
    citation: str
    expected_bytes: int
    adapter: str
    external_benchmark: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


REGISTRY: dict[str, DatasetRecord] = {
    "deepmind_mathematics": DatasetRecord(
        name="DeepMind Mathematics Dataset",
        url="https://github.com/google-deepmind/mathematics_dataset",
        version="1.0.1",
        license="Apache-2.0",
        citation="Saxton et al., Analysing Mathematical Reasoning Abilities of Neural Models, 2019",
        expected_bytes=2_000_000_000,
        adapter="DeepMindMathematicsAdapter",
    ),
    "srsd_feynman": DatasetRecord(
        name="SRSD-Feynman",
        url="https://github.com/omron-sinicx/srsd-benchmark",
        version="repository main / dataset release",
        license="CC-BY-4.0 data; MIT code",
        citation="Matsubara et al., Rethinking Symbolic Regression Datasets and Benchmarks, 2024",
        expected_bytes=1_200_000_000,
        adapter="SRSDFeynmanAdapter",
    ),
    "scibench": DatasetRecord(
        name="SciBench",
        url="https://github.com/mandyyyyii/scibench",
        version="main (2024 dataset revision)",
        license="MIT (verify source-text rights before redistribution)",
        citation="Wang et al., SciBench, ICML 2024",
        expected_bytes=250_000_000,
        adapter="SciBenchAdapter",
        external_benchmark=True,
    ),
    "tiny": DatasetRecord(
        name="Generated TinyDataset",
        url="generated-locally",
        version="1",
        license="MIT",
        citation="Fly Abstraction Lab synthetic smoke-test data",
        expected_bytes=100_000,
        adapter="TinyDatasetAdapter",
    ),
}


def get_dataset(name: str) -> DatasetRecord:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown dataset {name!r}; choices: {choices}") from exc
