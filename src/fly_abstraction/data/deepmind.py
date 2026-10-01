"""Safe local preparation of a finite-numeric DeepMind Mathematics subset."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tarfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, TextIO

from fly_abstraction.data.schema import MathProblem
from fly_abstraction.utils import sha256_file, write_json

VERSION = "1.0"
SOURCE_COMMIT = "427f45075f84b8b9774950196ad63867ca20ffb3"
SOURCE_URL = "https://console.cloud.google.com/storage/browser/mathematics-dataset"
REPOSITORY_URL = "https://github.com/google-deepmind/mathematics_dataset"
LICENSE = "Apache-2.0"
OFFICIAL_SPLITS = (
    "train-easy",
    "train-medium",
    "train-hard",
    "interpolate",
    "extrapolate",
)


def _safe_name(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member path: {name!r}")
    return path


def _split_and_module(name: str) -> tuple[str, str] | None:
    path = _safe_name(name)
    for index, part in enumerate(path.parts):
        if part in OFFICIAL_SPLITS and index + 1 < len(path.parts):
            module = "/".join(path.parts[index + 1 :])
            for suffix in (".txt", ".jsonl"):
                if module.endswith(suffix):
                    module = module[: -len(suffix)]
            return part, module
    return None


@contextmanager
def _text_reader(binary: BinaryIO) -> Iterator[TextIO]:
    wrapper = io.TextIOWrapper(binary, encoding="utf-8", errors="strict", newline=None)
    try:
        yield wrapper
    finally:
        wrapper.detach()


def _iter_sources(source: Path) -> Iterator[tuple[str, TextIO]]:
    if source.is_dir():
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source).as_posix()
            with path.open("r", encoding="utf-8", newline=None) as handle:
                yield relative, handle
        return
    suffixes = "".join(source.suffixes).lower()
    if suffixes.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(source, "r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                _safe_name(member.name)
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                with extracted, _text_reader(extracted) as handle:
                    yield member.name, handle
        return
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                if info.is_dir():
                    continue
                _safe_name(info.filename)
                with archive.open(info) as binary, _text_reader(binary) as handle:
                    yield info.filename, handle
        return
    raise ValueError("DeepMind --source must be a directory, .tar, .tar.gz, .tgz, or .zip")


def _source_sha256(source: Path) -> str:
    if source.is_file():
        return sha256_file(source)
    digest = hashlib.sha256()
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _pairs(handle: TextIO) -> Iterator[tuple[int, str, str]]:
    pair_index = 0
    while True:
        question = handle.readline()
        if question == "":
            return
        answer = handle.readline()
        if answer == "":
            raise ValueError("Source file has an unmatched final question line")
        yield pair_index, question.rstrip("\r\n"), answer.rstrip("\r\n")
        pair_index += 1


def _numeric_reason(answer: str) -> tuple[float | None, str | None]:
    try:
        value = float(answer.strip())
    except ValueError:
        return None, "not_plain_numeric"
    if not math.isfinite(value):
        return None, "non_finite_numeric"
    return value, None


def prepare_deepmind_numeric(source: Path, output_dir: Path) -> Path:
    """Filter a local official v1.0 archive while preserving every official split."""
    if not source.exists():
        raise FileNotFoundError(f"Local official DeepMind source is missing: {source}")
    if source.is_dir() and output_dir.resolve().is_relative_to(source.resolve()):
        raise ValueError("DeepMind output directory cannot be inside the source directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_path = output_dir / "rejected.jsonl"
    manifest_path = output_dir / "manifest.json"
    partials: dict[str, Path] = {}
    handles: dict[str, TextIO] = {}
    accepted: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    modules: dict[str, set[str]] = defaultdict(set)
    recognized_files = 0
    rejected_handle = rejected_path.with_suffix(".jsonl.part").open(
        "w", encoding="utf-8", newline="\n"
    )
    try:
        for relative_name, handle in _iter_sources(source):
            identity = _split_and_module(relative_name)
            if identity is None:
                continue
            split, module = identity
            recognized_files += 1
            modules[split].add(module)
            if split not in handles:
                partial = output_dir / f"{split}.jsonl.part"
                partials[split] = partial
                handles[split] = partial.open("w", encoding="utf-8", newline="\n")
            for pair_index, question, answer in _pairs(handle):
                _value, reason = _numeric_reason(answer)
                stable = f"{split}\0{module}\0{pair_index}\0{question}\0{answer}".encode()
                source_id = "deepmind-" + hashlib.sha256(stable).hexdigest()[:24]
                if reason is not None:
                    rejected[split] += 1
                    reasons[split][reason] += 1
                    rejected_handle.write(
                        json.dumps(
                            {
                                "source_id": source_id,
                                "official_split": split,
                                "module": module,
                                "record_index": pair_index,
                                "answer": answer,
                                "reason": reason,
                            },
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue
                problem = MathProblem(
                    source_id=source_id,
                    template_id=f"deepmind:{module}",
                    prompt=question,
                    answer=answer.strip(),
                    expression="numeric_target_unexposed",
                    split=split,
                    metadata={
                        "dataset": "deepmind_mathematics",
                        "dataset_version": VERSION,
                        "source_commit": SOURCE_COMMIT,
                        "official_split": split,
                        "module": module,
                        "record_index": pair_index,
                    },
                )
                handles[split].write(
                    json.dumps(problem.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
                )
                accepted[split] += 1
        if not recognized_files:
            raise ValueError(
                "No official DeepMind split files were found; expected paths containing "
                + ", ".join(OFFICIAL_SPLITS)
            )
        if not sum(accepted.values()):
            raise ValueError("DeepMind source contains no accepted finite numeric answers")
    except BaseException:
        for handle in handles.values():
            handle.close()
        rejected_handle.close()
        for partial in partials.values():
            partial.unlink(missing_ok=True)
        rejected_path.with_suffix(".jsonl.part").unlink(missing_ok=True)
        raise
    for handle in handles.values():
        handle.close()
    rejected_handle.close()
    outputs: dict[str, dict[str, Any]] = {}
    for split, partial in sorted(partials.items()):
        final = output_dir / f"{split}.jsonl"
        os.replace(partial, final)
        outputs[split] = {
            "path": str(final),
            "sha256": sha256_file(final),
            "accepted_records": accepted[split],
            "rejected_records": rejected[split],
            "rejection_reasons": dict(sorted(reasons[split].items())),
            "modules": sorted(modules[split]),
        }
    os.replace(rejected_path.with_suffix(".jsonl.part"), rejected_path)
    write_json(
        manifest_path,
        {
            "dataset": "deepmind_mathematics",
            "version": VERSION,
            "source_commit": SOURCE_COMMIT,
            "source_url": SOURCE_URL,
            "repository_url": REPOSITORY_URL,
            "license": LICENSE,
            "source_path": str(source),
            "source_sha256": _source_sha256(source),
            "filter": "answer parses directly as a finite Python float",
            "official_splits_kept_separate": True,
            "outputs": outputs,
            "accepted_records": sum(accepted.values()),
            "rejected_records": sum(rejected.values()),
            "rejection_reasons": dict(
                sorted(sum((counter for counter in reasons.values()), Counter()).items())
            ),
            "rejected_manifest": str(rejected_path),
            "rejected_manifest_sha256": sha256_file(rejected_path),
        },
    )
    return manifest_path
