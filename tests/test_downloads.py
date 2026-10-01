import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

from fly_abstraction.data import downloads
from fly_abstraction.data.downloads import download_registered_dataset
from fly_abstraction.data.registry import get_dataset


def test_confirmed_download_streams_and_records_sha256(tmp_path: Path, monkeypatch) -> None:
    payload = b"X1,X2,X3,X4,X5,X6,X7,X8,Y1,Y2\n1,2,3,4,5,6,7,8,9,10\n"
    expected = hashlib.sha256(payload).hexdigest()
    record = replace(
        get_dataset("uci_energy_efficiency"),
        name="Test UCI",
        expected_bytes=len(payload),
        published_checksum=f"sha256:{expected}",
    )
    monkeypatch.setattr(downloads, "urlopen", lambda *_args, **_kwargs: io.BytesIO(payload))
    path, manifest_path = download_registered_dataset(record, tmp_path)
    assert path.read_bytes() == payload
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sha256"] == expected
    assert manifest["version"] == record.version
    assert manifest["accepted_records"] == 0
    assert manifest["rejection_reasons"] == {}
