"""Deterministic preparation of a bounded FlyWire FAFB v783 graph core."""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import os
import subprocess
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import torch

from fly_abstraction import __version__
from fly_abstraction.data.registry import get_dataset
from fly_abstraction.graph.connectome import ConnectomeGraph, load_graph_npz, save_graph_npz
from fly_abstraction.utils import sha256_file, write_json

REQUIRED_COLUMNS = ("pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count")
ALGORITHM = "weighted_connected_core_v1"
RELEASE = "FAFB v783"


def preparation_plan(
    source: Path,
    output: Path,
    *,
    nodes: int,
    min_pair_synapses: float,
    input_node_count: int,
    output_node_count: int,
    seed: int,
) -> dict[str, Any]:
    """Validate a preparation request and return a side-effect-free resource preview."""
    _validate_arguments(
        source,
        output,
        nodes=nodes,
        min_pair_synapses=min_pair_synapses,
        input_node_count=input_node_count,
        output_node_count=output_node_count,
    )
    schema = _validate_feather_schema(source)
    source_bytes = source.stat().st_size
    estimated_peak_ram = source_bytes * 4 + 512 * 1024**2
    return {
        "dataset": "flywire_fafb_v783",
        "release": RELEASE,
        "source": source.as_posix(),
        "source_bytes": source_bytes,
        "output": output.as_posix(),
        "schema": schema,
        "selection_algorithm": ALGORITHM,
        "selection": (
            "global maximum total strength, then undirected frontier maximum total strength; "
            "ties use ascending numeric root_id; disconnected components restart by global rank"
        ),
        "parameters": {
            "nodes": nodes,
            "min_pair_synapses": min_pair_synapses,
            "input_node_count": input_node_count,
            "output_node_count": output_node_count,
            "seed": seed,
        },
        "estimated_peak_ram_bytes": estimated_peak_ram,
        "available_ram_bytes": psutil.virtual_memory().available,
    }


def prepare_flywire_v783(
    source: Path,
    output: Path,
    *,
    nodes: int,
    min_pair_synapses: float,
    input_node_count: int,
    output_node_count: int,
    seed: int,
) -> dict[str, Path]:
    """Prepare a deterministic directed induced core from the official Feather table."""
    plan = preparation_plan(
        source,
        output,
        nodes=nodes,
        min_pair_synapses=min_pair_synapses,
        input_node_count=input_node_count,
        output_node_count=output_node_count,
        seed=seed,
    )
    if int(plan["estimated_peak_ram_bytes"]) > int(plan["available_ram_bytes"]) * 0.8:
        raise RuntimeError(
            "Estimated preparation RAM exceeds 80% of currently available RAM; "
            "close applications or use a host with more memory"
        )

    nodes_path = output.with_suffix(".nodes.csv")
    manifest_path = output.with_suffix(".manifest.json")
    stats_path = output.with_suffix(".stats.json")
    targets = (output, nodes_path, manifest_path, stats_path)
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite FlyWire artifacts: " + ", ".join(existing))

    source_sha256 = sha256_file(source)
    download_manifest = _load_download_manifest(source, source_sha256)
    try:
        arrays, row_stats = _read_valid_rows(source)
    except pa.ArrowException as exc:
        raise ValueError(f"Cannot scan FlyWire Feather data: {exc}") from exc
    pair_source, pair_target, pair_weight = _aggregate_pairs(*arrays)
    del arrays
    original_unique_pairs = int(pair_weight.size)
    threshold_mask = pair_weight >= min_pair_synapses
    pair_source = pair_source[threshold_mask]
    pair_target = pair_target[threshold_mask]
    pair_weight = pair_weight[threshold_mask]
    pairs_after_threshold = int(pair_weight.size)
    if not pairs_after_threshold:
        raise ValueError("No directed pairs remain after min-pair-synapses thresholding")

    root_ids = np.unique(np.concatenate((pair_source, pair_target))).astype(np.int64)
    if root_ids.size < nodes:
        raise ValueError(f"Only {root_ids.size} nodes remain after thresholding; requested {nodes}")
    source_index = np.searchsorted(root_ids, pair_source)
    target_index = np.searchsorted(root_ids, pair_target)
    incoming = np.bincount(target_index, weights=pair_weight, minlength=root_ids.size)
    outgoing = np.bincount(source_index, weights=pair_weight, minlength=root_ids.size)
    total = incoming + outgoing
    selected_global, selected_components = _weighted_connected_core(
        root_ids, source_index, target_index, total, nodes
    )
    selected_mask = np.zeros(root_ids.size, dtype=bool)
    selected_mask[selected_global] = True
    induced_mask = selected_mask[source_index] & selected_mask[target_index]
    induced_source_global = source_index[induced_mask]
    induced_target_global = target_index[induced_mask]
    induced_weight = pair_weight[induced_mask]

    global_to_core = np.full(root_ids.size, -1, dtype=np.int64)
    global_to_core[selected_global] = np.arange(nodes, dtype=np.int64)
    core_source = global_to_core[induced_source_global]
    core_target = global_to_core[induced_target_global]
    order = np.lexsort((core_target, core_source))
    core_source = core_source[order]
    core_target = core_target[order]
    core_weight = induced_weight[order]
    selected_root_ids = root_ids[selected_global]

    core_incoming = np.bincount(core_target, weights=core_weight, minlength=nodes)
    core_outgoing = np.bincount(core_source, weights=core_weight, minlength=nodes)
    input_nodes, output_nodes = _build_interface_mappings(
        selected_root_ids,
        core_incoming,
        core_outgoing,
        input_node_count,
        output_node_count,
    )
    reachable_fraction = _reachable_output_fraction(
        nodes, core_source, core_target, input_nodes, output_nodes
    )
    selected_root_ids_sha256 = _int_array_sha256(selected_root_ids)
    graph_manifest = {
        "dataset": "flywire_fafb_v783",
        "release": RELEASE,
        "doi": "10.5281/zenodo.10676866",
        "source_url": get_dataset("flywire_fafb_v783").download_url,
        "source_sha256": source_sha256,
        "algorithm": ALGORITHM,
        "algorithm_version": 1,
        "parameters": dict(plan["parameters"]),
        "accepted_records": row_stats["accepted_records"],
        "rejected_records": row_stats["rejected_records"],
        "rejection_reasons": row_stats["rejection_reasons"],
        "original_unique_pairs": original_unique_pairs,
        "pairs_after_threshold": pairs_after_threshold,
        "requested_nodes": nodes,
        "actual_nodes": nodes,
        "directed_edges": int(core_weight.size),
        "self_loops": int(np.count_nonzero(core_source == core_target)),
        "selection_component_count": selected_components,
        "input_nodes": input_nodes.tolist(),
        "output_nodes": output_nodes.tolist(),
        "input_root_ids": selected_root_ids[input_nodes].tolist(),
        "output_root_ids": selected_root_ids[output_nodes].tolist(),
        "selected_root_ids_sha256": selected_root_ids_sha256,
        "reachable_output_fraction_from_inputs": reachable_fraction,
    }
    graph = ConnectomeGraph(
        num_nodes=nodes,
        edge_index=torch.from_numpy(np.stack((core_source, core_target))).long(),
        edge_weight=torch.from_numpy(core_weight.astype(np.float32, copy=False)),
        input_nodes=torch.from_numpy(input_nodes.copy()).long(),
        output_nodes=torch.from_numpy(output_nodes.copy()).long(),
        metadata={
            "name": "flywire_fafb_v783_weighted_connected_core",
            "directed": True,
            "weighted": True,
            "interface": "artificial task-independent strength-ranked mapping",
        },
        manifest=graph_manifest,
    )

    created: list[Path] = []
    try:
        _write_nodes_csv(
            nodes_path,
            selected_root_ids,
            core_incoming,
            core_outgoing,
            input_nodes,
            output_nodes,
        )
        created.append(nodes_path)
        save_graph_npz(output, graph)
        created.append(output)
        stats = inspect_graph(output, validate_sidecar=False)
        write_json(stats_path, stats)
        created.append(stats_path)
        manifest = {
            **graph_manifest,
            "source": source.as_posix(),
            "output": output.as_posix(),
            "cli_parameters": {
                "source": source.as_posix(),
                "output": output.as_posix(),
                **plan["parameters"],
                "confirm_prepare": True,
            },
            "license": get_dataset("flywire_fafb_v783").license,
            "terms_url": get_dataset("flywire_fafb_v783").terms_url,
            "download_manifest": download_manifest,
            "nodes_csv": nodes_path.as_posix(),
            "stats_json": stats_path.as_posix(),
            "npz_sha256": sha256_file(output),
            "nodes_csv_sha256": sha256_file(nodes_path),
            "stats_json_sha256": sha256_file(stats_path),
            "graph_content_sha256": graph.content_hash(),
            "git_commit": _git_commit(),
            "software": {
                "fly_abstraction": __version__,
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
                "torch": torch.__version__,
            },
        }
        write_json(manifest_path, manifest)
        created.append(manifest_path)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return {
        "graph": output,
        "nodes": nodes_path,
        "manifest": manifest_path,
        "stats": stats_path,
    }


def _validate_arguments(
    source: Path,
    output: Path,
    *,
    nodes: int,
    min_pair_synapses: float,
    input_node_count: int,
    output_node_count: int,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"FlyWire Feather source not found: {source}")
    if source.suffix.lower() != ".feather":
        raise ValueError("FlyWire source must use the .feather suffix")
    if output.suffix.lower() != ".npz":
        raise ValueError("FlyWire output must use the .npz suffix")
    if nodes <= 0:
        raise ValueError("--nodes must be positive")
    if not np.isfinite(min_pair_synapses) or min_pair_synapses <= 0:
        raise ValueError("--min-pair-synapses must be finite and positive")
    for name, count in (
        ("--input-node-count", input_node_count),
        ("--output-node-count", output_node_count),
    ):
        if count <= 0 or count > nodes:
            raise ValueError(f"{name} must be in 1..nodes")
    for path in (
        output,
        output.with_suffix(".nodes.csv"),
        output.with_suffix(".manifest.json"),
        output.with_suffix(".stats.json"),
    ):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")


def _validate_feather_schema(source: Path) -> dict[str, str]:
    try:
        schema = pads.dataset(source, format="ipc").schema
    except pa.ArrowException as exc:
        raise ValueError(f"Cannot read FlyWire Feather schema: {exc}") from exc
    missing = [name for name in REQUIRED_COLUMNS if name not in schema.names]
    if missing:
        raise ValueError(f"FlyWire Feather is missing required columns: {missing}")
    pre_type = schema.field("pre_pt_root_id").type
    post_type = schema.field("post_pt_root_id").type
    syn_type = schema.field("syn_count").type
    neuropil_type = schema.field("neuropil").type
    if not pa.types.is_integer(pre_type) or not pa.types.is_integer(post_type):
        raise ValueError("pre_pt_root_id and post_pt_root_id must be integer columns")
    if not pa.types.is_integer(syn_type) and not pa.types.is_floating(syn_type):
        raise ValueError("syn_count must be numeric")
    neuropil_is_text = pa.types.is_string(neuropil_type) or pa.types.is_large_string(neuropil_type)
    if pa.types.is_dictionary(neuropil_type):
        neuropil_is_text = pa.types.is_string(neuropil_type.value_type) or pa.types.is_large_string(
            neuropil_type.value_type
        )
    if not neuropil_is_text:
        raise ValueError("neuropil must be a string or dictionary-encoded string column")
    return {name: str(schema.field(name).type) for name in REQUIRED_COLUMNS}


def _load_download_manifest(source: Path, source_sha256: str) -> dict[str, Any]:
    path = source.with_suffix(source.suffix + ".download.json")
    if not path.is_file():
        return {"status": "not present; source SHA256 computed locally"}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("sha256") != source_sha256:
        raise ValueError("FlyWire source SHA256 does not match its adjacent download manifest")
    return {
        "status": "verified",
        "path": path.as_posix(),
        "sha256": source_sha256,
        "bytes": payload.get("bytes"),
        "source_url": payload.get("source_url"),
        "doi": payload.get("doi"),
        "published_checksum": payload.get("published_checksum"),
        "published_checksum_verified": payload.get("published_checksum_verified"),
    }


def _read_valid_rows(
    source: Path,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    dataset = pads.dataset(source, format="ipc")
    scanner = dataset.scanner(
        columns=list(REQUIRED_COLUMNS),
        batch_size=262_144,
        use_threads=True,
    )
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    accepted = 0
    rejected = 0
    reasons: Counter[str] = Counter()
    for batch in scanner.to_batches():
        try:
            pre_array = pc.cast(batch.column(0), pa.int64(), safe=True)
            post_array = pc.cast(batch.column(1), pa.int64(), safe=True)
            neuropil_array = pc.cast(batch.column(2), pa.string(), safe=True)
            syn_array = pc.cast(batch.column(3), pa.float64(), safe=True)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            raise ValueError(
                f"FlyWire Feather contains values incompatible with its schema: {exc}"
            ) from exc

        pre = pc.fill_null(pre_array, 0).to_numpy(zero_copy_only=False)
        post = pc.fill_null(post_array, 0).to_numpy(zero_copy_only=False)
        syn = pc.fill_null(syn_array, np.nan).to_numpy(zero_copy_only=False)
        neuropil_empty = pc.equal(
            pc.utf8_trim_whitespace(pc.fill_null(neuropil_array, "")), ""
        ).to_numpy(zero_copy_only=False)
        remaining = np.ones(batch.num_rows, dtype=bool)
        invalid = remaining & (pre <= 0)
        reasons["empty_or_invalid_pre_pt_root_id"] += int(np.count_nonzero(invalid))
        remaining &= ~invalid
        invalid = remaining & (post <= 0)
        reasons["empty_or_invalid_post_pt_root_id"] += int(np.count_nonzero(invalid))
        remaining &= ~invalid
        invalid = remaining & (~np.isfinite(syn) | (syn <= 0))
        reasons["non_finite_or_non_positive_syn_count"] += int(np.count_nonzero(invalid))
        remaining &= ~invalid
        invalid = remaining & neuropil_empty
        reasons["empty_neuropil"] += int(np.count_nonzero(invalid))
        remaining &= ~invalid
        batch_accepted = int(np.count_nonzero(remaining))
        accepted += batch_accepted
        rejected += batch.num_rows - batch_accepted
        if batch_accepted:
            source_parts.append(np.asarray(pre[remaining], dtype=np.int64))
            target_parts.append(np.asarray(post[remaining], dtype=np.int64))
            weight_parts.append(np.asarray(syn[remaining], dtype=np.float64))
    if not accepted:
        raise ValueError("FlyWire preparation rejected every input row")
    return (
        (
            np.concatenate(source_parts),
            np.concatenate(target_parts),
            np.concatenate(weight_parts),
        ),
        {
            "accepted_records": accepted,
            "rejected_records": rejected,
            "rejection_reasons": {key: value for key, value in sorted(reasons.items()) if value},
        },
    )


def _aggregate_pairs(
    source: np.ndarray, target: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Weight is a tertiary key so floating additions are independent of source row order.
    order = np.lexsort((weight, target, source))
    source = source[order]
    target = target[order]
    weight = weight[order]
    starts = np.empty(source.size, dtype=bool)
    starts[0] = True
    starts[1:] = (source[1:] != source[:-1]) | (target[1:] != target[:-1])
    indices = np.flatnonzero(starts)
    return source[indices], target[indices], np.add.reduceat(weight, indices)


def _weighted_connected_core(
    root_ids: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    total_strength: np.ndarray,
    requested_nodes: int,
) -> tuple[np.ndarray, int]:
    undirected_source = np.concatenate((source, target))
    undirected_target = np.concatenate((target, source))
    order = np.lexsort((undirected_target, undirected_source))
    undirected_source = undirected_source[order]
    undirected_target = undirected_target[order]
    offsets = np.zeros(root_ids.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.bincount(undirected_source, minlength=root_ids.size), dtype=np.int64)
    global_rank = np.lexsort((root_ids, -total_strength))
    selected = np.zeros(root_ids.size, dtype=bool)
    queued = np.zeros(root_ids.size, dtype=bool)
    selected_order: list[int] = []
    frontier: list[tuple[float, int, int]] = []
    component_count = 0
    global_cursor = 0

    def add_node(index: int) -> None:
        selected[index] = True
        selected_order.append(index)
        for position in range(int(offsets[index]), int(offsets[index + 1])):
            neighbor = int(undirected_target[position])
            if not selected[neighbor] and not queued[neighbor]:
                queued[neighbor] = True
                heapq.heappush(
                    frontier,
                    (-float(total_strength[neighbor]), int(root_ids[neighbor]), neighbor),
                )

    while len(selected_order) < requested_nodes:
        while frontier and selected[frontier[0][2]]:
            heapq.heappop(frontier)
        if frontier:
            _, _, candidate = heapq.heappop(frontier)
            if selected[candidate]:
                continue
            add_node(candidate)
            continue
        while global_cursor < global_rank.size and selected[global_rank[global_cursor]]:
            global_cursor += 1
        if global_cursor >= global_rank.size:
            break
        component_count += 1
        start = int(global_rank[global_cursor])
        queued[start] = True
        add_node(start)

    if len(selected_order) != requested_nodes:
        raise ValueError(
            f"Connected-core selection produced {len(selected_order)} of {requested_nodes} nodes"
        )
    return np.asarray(selected_order, dtype=np.int64), component_count


def _build_interface_mappings(
    root_ids: np.ndarray,
    incoming: np.ndarray,
    outgoing: np.ndarray,
    input_count: int,
    output_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    input_rank = np.lexsort((root_ids, -outgoing))
    input_nodes = input_rank[:input_count].astype(np.int64)
    input_mask = np.zeros(root_ids.size, dtype=bool)
    input_mask[input_nodes] = True
    output_rank = np.lexsort((root_ids, -incoming))
    non_input = output_rank[~input_mask[output_rank]]
    if non_input.size >= output_count:
        output_nodes = non_input[:output_count]
    else:
        input_candidates = output_rank[input_mask[output_rank]]
        output_nodes = np.concatenate(
            (non_input, input_candidates[: output_count - non_input.size])
        )
    return input_nodes, output_nodes.astype(np.int64)


def _reachable_output_fraction(
    num_nodes: int,
    source: np.ndarray,
    target: np.ndarray,
    input_nodes: np.ndarray,
    output_nodes: np.ndarray,
) -> float:
    order = np.lexsort((target, source))
    ordered_source = source[order]
    ordered_target = target[order]
    offsets = np.zeros(num_nodes + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.bincount(ordered_source, minlength=num_nodes), dtype=np.int64)
    reached = np.zeros(num_nodes, dtype=bool)
    queue: deque[int] = deque()
    for node in input_nodes:
        index = int(node)
        if not reached[index]:
            reached[index] = True
            queue.append(index)
    while queue:
        node = queue.popleft()
        for position in range(int(offsets[node]), int(offsets[node + 1])):
            neighbor = int(ordered_target[position])
            if not reached[neighbor]:
                reached[neighbor] = True
                queue.append(neighbor)
    return float(np.count_nonzero(reached[output_nodes]) / output_nodes.size)


def _weak_component_count(num_nodes: int, source: np.ndarray, target: np.ndarray) -> int:
    if not source.size:
        return num_nodes
    undirected_source = np.concatenate((source, target))
    undirected_target = np.concatenate((target, source))
    order = np.lexsort((undirected_target, undirected_source))
    undirected_source = undirected_source[order]
    undirected_target = undirected_target[order]
    offsets = np.zeros(num_nodes + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.bincount(undirected_source, minlength=num_nodes), dtype=np.int64)
    visited = np.zeros(num_nodes, dtype=bool)
    components = 0
    for start in range(num_nodes):
        if visited[start]:
            continue
        components += 1
        visited[start] = True
        queue: deque[int] = deque((start,))
        while queue:
            node = queue.popleft()
            for position in range(int(offsets[node]), int(offsets[node + 1])):
                neighbor = int(undirected_target[position])
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
    return components


def _summary(values: np.ndarray) -> dict[str, float | int]:
    if not values.size:
        return {"min": 0, "median": 0.0, "max": 0}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def inspect_graph(path: Path, *, validate_sidecar: bool = True) -> dict[str, Any]:
    """Return JSON-serializable graph statistics without mutating any artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"Prepared graph not found: {path}")
    graph = load_graph_npz(path)
    source = graph.edge_index[0].cpu().numpy()
    target = graph.edge_index[1].cpu().numpy()
    weight = graph.edge_weight.cpu().numpy()
    input_nodes = graph.input_nodes.cpu().numpy()
    output_nodes = graph.output_nodes.cpu().numpy()
    in_degree = np.bincount(target, minlength=graph.num_nodes)
    out_degree = np.bincount(source, minlength=graph.num_nodes)
    npz_sha256 = sha256_file(path)
    sidecar_path = path.with_suffix(".manifest.json")
    if validate_sidecar and sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8-sig"))
        expected = sidecar.get("npz_sha256")
        if expected and expected != npz_sha256:
            raise ValueError("Prepared graph SHA256 does not match its sidecar manifest")
    return {
        "path": path.as_posix(),
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "density": float(graph.num_edges / (graph.num_nodes * graph.num_nodes)),
        "self_loops": int(np.count_nonzero(source == target)),
        "input_node_count": int(input_nodes.size),
        "output_node_count": int(output_nodes.size),
        "in_degree": _summary(in_degree),
        "out_degree": _summary(out_degree),
        "edge_weight": _summary(weight),
        "weak_component_count": _weak_component_count(graph.num_nodes, source, target),
        "reachable_output_fraction_from_inputs": _reachable_output_fraction(
            graph.num_nodes, source, target, input_nodes, output_nodes
        ),
        "release": graph.manifest.get("release"),
        "algorithm": graph.manifest.get("algorithm"),
        "source_sha256": graph.manifest.get("source_sha256"),
        "content_hash": graph.content_hash(),
        "npz_sha256": npz_sha256,
    }


def _write_nodes_csv(
    path: Path,
    root_ids: np.ndarray,
    incoming: np.ndarray,
    outgoing: np.ndarray,
    input_nodes: np.ndarray,
    output_nodes: np.ndarray,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite node table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite partial node table: {temporary}")
    input_set = set(int(value) for value in input_nodes)
    output_set = set(int(value) for value in output_nodes)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator=chr(10))
            writer.writerow(
                (
                    "node_index",
                    "root_id",
                    "incoming_strength",
                    "outgoing_strength",
                    "total_strength",
                    "is_input",
                    "is_output",
                )
            )
            for index, root_id in enumerate(root_ids):
                writer.writerow(
                    (
                        index,
                        int(root_id),
                        format(float(incoming[index]), ".17g"),
                        format(float(outgoing[index]), ".17g"),
                        format(float(incoming[index] + outgoing[index]), ".17g"),
                        int(index in input_set),
                        int(index in output_set),
                    )
                )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _int_array_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    canonical = np.asarray(values, dtype="<i8")
    raw = memoryview(canonical).cast("B")
    for offset in range(0, len(raw), 1024 * 1024):
        digest.update(raw[offset : offset + 1024 * 1024])
    return digest.hexdigest()


def _git_commit() -> str:
    repository = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"
