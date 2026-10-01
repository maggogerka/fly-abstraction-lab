"""Interface boundary for future local FlyWire FAFB v783 artifacts."""

from __future__ import annotations

import csv
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from fly_abstraction.graph.connectome import ConnectomeGraph, load_graph_npz, save_graph_npz
from fly_abstraction.utils import sha256_file


class ConnectomeAdapter(ABC):
    @abstractmethod
    def load(self, path: Path) -> ConnectomeGraph:
        """Load an already-downloaded graph artifact."""


class FlyWireFAFBV783Adapter(ConnectomeAdapter):
    """Loads a prepared v783 NPZ export; it deliberately performs no network access."""

    release = "FAFB v783"

    def load(self, path: Path) -> ConnectomeGraph:
        if not path.is_file():
            raise FileNotFoundError(
                f"Prepared {self.release} artifact not found at {path}; downloads require "
                "explicit consent"
            )
        if path.suffix.lower() != ".npz":
            raise ValueError("FlyWire graphs must use the prepared .npz binary format")
        graph = load_graph_npz(path)
        manifest_release = graph.manifest.get("release")
        if manifest_release != self.release:
            raise ValueError(f"Expected release {self.release!r}, got {manifest_release!r}")
        return graph


def convert_local_flywire_export(edges_csv: Path, output: Path) -> Path:
    """Aggregate an authorized local export by pre_group/post_group into an NPZ graph.

    This function never accesses FlyWire. The caller is responsible for export authorization,
    release provenance, and compliance with the applicable data-use terms.
    """
    if not edges_csv.is_file():
        raise FileNotFoundError(f"Local FlyWire export is missing: {edges_csv}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing graph: {output}")
    node_map_path = output.with_suffix(".nodes.csv")
    if node_map_path.exists():
        raise FileExistsError(f"Refusing to overwrite node mapping: {node_map_path}")
    database = output.with_suffix(".aggregate.sqlite.tmp")
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "CREATE TABLE edges (source TEXT, target TEXT, weight REAL, "
            "PRIMARY KEY (source, target))"
        )
        with edges_csv.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"pre_group", "post_group", "synapse_count"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"FlyWire CSV must contain columns {sorted(required)}")
            batch: list[tuple[str, str, float]] = []
            for row in reader:
                weight = float(row["synapse_count"])
                if weight <= 0:
                    raise ValueError("synapse_count values must be positive")
                batch.append((row["pre_group"], row["post_group"], weight))
                if len(batch) == 10_000:
                    _upsert_edges(connection, batch)
                    batch.clear()
            _upsert_edges(connection, batch)
        nodes = [
            row[0]
            for row in connection.execute(
                "SELECT source FROM edges UNION SELECT target FROM edges ORDER BY 1"
            )
        ]
        node_to_index = {node: index for index, node in enumerate(nodes)}
        edge_count = int(connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
        edge_index = torch.empty((2, edge_count), dtype=torch.long)
        edge_weight = torch.empty(edge_count, dtype=torch.float32)
        for index, (source, target, weight) in enumerate(
            connection.execute("SELECT source, target, weight FROM edges ORDER BY source, target")
        ):
            edge_index[0, index] = node_to_index[source]
            edge_index[1, index] = node_to_index[target]
            edge_weight[index] = weight
        graph = ConnectomeGraph(
            num_nodes=len(nodes),
            edge_index=edge_index,
            edge_weight=edge_weight,
            metadata={"name": "flywire_fafb_v783_aggregated", "directed": True},
            manifest={
                "release": "FAFB v783",
                "source": "authorized user-local export",
                "source_sha256": sha256_file(edges_csv),
                "aggregation": "sum synapse_count by (pre_group, post_group)",
            },
        )
        save_graph_npz(output, graph)
        node_map_path.parent.mkdir(parents=True, exist_ok=True)
        with node_map_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("node_index", "group_id"))
            writer.writerows(enumerate(nodes))
    finally:
        connection.close()
        database.unlink(missing_ok=True)
    return output


def _upsert_edges(connection: sqlite3.Connection, rows: list[tuple[str, str, float]]) -> None:
    if not rows:
        return
    connection.executemany(
        "INSERT INTO edges(source, target, weight) VALUES (?, ?, ?) "
        "ON CONFLICT(source, target) DO UPDATE SET weight = weight + excluded.weight",
        rows,
    )
    connection.commit()
