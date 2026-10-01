"""Consent-gated, streaming downloads from registry-pinned official URLs."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from fly_abstraction.data.registry import DatasetRecord


def download_registered_dataset(record: DatasetRecord, root: Path) -> tuple[Path, Path]:
    if not record.download_url or not record.filename:
        raise ValueError(f"{record.name} has no approved automatic download")
    destination_dir = root / record.name.lower().replace(" ", "_")
    destination = destination_dir / record.filename
    manifest_path = destination.with_suffix(destination.suffix + ".download.json")
    if destination.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing download: {destination}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    manifest_temporary = manifest_path.with_suffix(manifest_path.suffix + ".part")
    if temporary.exists() or manifest_temporary.exists():
        raise FileExistsError("A partial download already exists; inspect it before retrying")
    digest = hashlib.sha256()
    size = 0
    destination_moved = False
    request = Request(record.download_url, headers={"User-Agent": "fly-abstraction-lab/0.1"})
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > max(record.expected_bytes * 20, 10 * 1024 * 1024):
                    raise RuntimeError("Download exceeded the registry safety size limit")
                digest.update(chunk)
                handle.write(chunk)
        sha256 = digest.hexdigest()
        if record.published_checksum:
            algorithm, expected = record.published_checksum.split(":", 1)
            if algorithm != "sha256" or sha256.lower() != expected.lower():
                raise RuntimeError("Downloaded dataset checksum does not match the registry")
        manifest = {
            "name": record.name,
            "version": record.version,
            "license": record.license,
            "source_url": record.download_url,
            "bytes": size,
            "sha256": sha256,
            "checksum_policy": record.checksum_policy,
            "downloaded_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest_temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
        destination_moved = True
        os.replace(manifest_temporary, manifest_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)
        if destination_moved and not manifest_path.exists():
            destination.unlink(missing_ok=True)
        raise
    return destination, manifest_path
