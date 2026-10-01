"""Interface boundary for future local FlyWire FAFB v783 artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from fly_abstraction.graph.connectome import ConnectomeGraph, load_graph_json


class ConnectomeAdapter(ABC):
    @abstractmethod
    def load(self, path: Path) -> ConnectomeGraph:
        """Load an already-downloaded graph artifact."""


class FlyWireFAFBV783Adapter(ConnectomeAdapter):
    """Loads a prepared v783 JSON export; it deliberately performs no network access."""

    release = "FAFB v783"

    def load(self, path: Path) -> ConnectomeGraph:
        if not path.is_file():
            raise FileNotFoundError(
                f"Prepared {self.release} artifact not found at {path}; downloads require "
                "explicit consent"
            )
        graph = load_graph_json(path)
        manifest_release = graph.manifest.get("release")
        if manifest_release != self.release:
            raise ValueError(f"Expected release {self.release!r}, got {manifest_release!r}")
        return graph
