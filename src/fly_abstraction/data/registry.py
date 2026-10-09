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
    download_url: str | None = None
    filename: str | None = None
    published_checksum: str | None = None
    checksum_policy: str = "metadata-only; no automatic download"
    storage_subdir: str | None = None
    doi: str | None = None
    terms_url: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


REGISTRY: dict[str, DatasetRecord] = {
    "deepmind_mathematics": DatasetRecord(
        name="DeepMind Mathematics Dataset",
        url="https://console.cloud.google.com/storage/browser/mathematics-dataset",
        version="1.0 @ source commit 427f45075f84b8b9774950196ad63867ca20ffb3",
        license="Apache-2.0",
        citation="Saxton et al., Analysing Mathematical Reasoning Abilities of Neural Models, 2019",
        expected_bytes=20_000_000_000,
        adapter="DeepMindMathematicsAdapter",
        checksum_policy=(
            "local official archive only; preparation computes SHA256 because no pinned "
            "official direct archive checksum is published in the repository README"
        ),
    ),
    "flywire_fafb_v783": DatasetRecord(
        name="FlyWire FAFB v783 proofread connections",
        url="https://zenodo.org/records/10676866",
        version="783.0",
        license="Not specified on the Zenodo record; source terms apply",
        citation="FlyWire Consortium, FlyWire Whole-brain Connectome Connectivity Data, 2024",
        expected_bytes=852_022_274,
        adapter="FlyWireFAFBV783Adapter",
        download_url=(
            "https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1"
        ),
        filename="proofread_connections_783.feather",
        published_checksum="md5:f48f972d262323a102aed49af1396b8a",
        checksum_policy=(
            "verify the Zenodo-published MD5 and always compute SHA256 during the "
            "confirmed streaming download"
        ),
        storage_subdir="flywire_fafb_v783",
        doi="10.5281/zenodo.10676866",
        terms_url="https://about.zenodo.org/terms/",
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
    "uci_energy_efficiency": DatasetRecord(
        name="UCI Energy Efficiency",
        url="https://archive.ics.uci.edu/dataset/242/energy+efficiency",
        version="UCI-242-2024-02-26",
        license="CC BY 4.0",
        citation="Tsanas and Xifara, Energy and Buildings 49 (2012), DOI 10.24432/C51307",
        expected_bytes=100_000,
        adapter="UCIEnergyEfficiencyAdapter",
        download_url="https://archive.ics.uci.edu/static/public/242/data.csv",
        filename="data.csv",
        published_checksum=None,
        checksum_policy=(
            "UCI does not publish a digest in its API; compute SHA256 during the confirmed "
            "download and pin it in the adjacent manifest"
        ),
    ),
}


def get_dataset(name: str) -> DatasetRecord:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown dataset {name!r}; choices: {choices}") from exc
