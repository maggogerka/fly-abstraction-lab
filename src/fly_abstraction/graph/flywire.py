"""Interface boundary for future local FlyWire FAFB v783 artifacts."""

from __future__ import annotations

import csv
import sqlite3
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

import torch

from fly_abstraction.graph.connectome import ConnectomeGraph, load_graph_npz, save_graph_npz
from fly_abstraction.utils import sha256_file, write_json


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
    manifest_path = output.with_suffix(".manifest.json")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite graph manifest: {manifest_path}")
    database = output.with_suffix(".aggregate.sqlite.tmp")
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    accepted_rows = 0
    rejected_rows = 0
    rejection_reasons: Counter[str] = Counter()
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
                try:
                    weight = float(row["synapse_count"])
                    if weight <= 0:
                        raise ValueError("non-positive synapse_count")
                    if not row["pre_group"].strip() or not row["post_group"].strip():
                        raise ValueError("empty group ID")
                except (KeyError, TypeError, ValueError) as exc:
                    rejected_rows += 1
                    rejection_reasons[str(exc)] += 1
                    continue
                accepted_rows += 1
                batch.append((row["pre_group"], row["post_group"], weight))
                if len(batch) == 10_000:
                    _upsert_edges(connection, batch)
                    batch.clear()
            _upsert_edges(connection, batch)
        if not accepted_rows:
            raise ValueError("FlyWire conversion rejected every edge row")
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
        provenance = {
            "release": "FAFB v783",
            "source": "authorized user-local export",
            "source_url": "local-only; governed by the user's FlyWire access terms",
            "version": "FAFB v783",
            "license": "not redistributed; verify the user's FlyWire terms",
            "source_sha256": sha256_file(edges_csv),
            "aggregation": "sum synapse_count by (pre_group, post_group)",
            "accepted_records": accepted_rows,
            "rejected_records": rejected_rows,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        }
        graph = ConnectomeGraph(
            num_nodes=len(nodes),
            edge_index=edge_index,
            edge_weight=edge_weight,
            metadata={"name": "flywire_fafb_v783_aggregated", "directed": True},
            manifest=provenance,
        )
        save_graph_npz(output, graph)
        node_map_path.parent.mkdir(parents=True, exist_ok=True)
        with node_map_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("node_index", "group_id"))
            writer.writerows(enumerate(nodes))
        write_json(
            manifest_path,
            {
                **provenance,
                "prepared_path": str(output),
                "prepared_sha256": sha256_file(output),
                "node_map_path": str(node_map_path),
                "node_map_sha256": sha256_file(node_map_path),
                "num_nodes": len(nodes),
                "num_edges": edge_count,
            },
        )
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
