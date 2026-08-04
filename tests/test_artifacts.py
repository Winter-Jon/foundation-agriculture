from pathlib import Path

import yaml

from agrinet.common.artifacts import register_file_artifact, sha256_file


def test_register_file_artifact_is_atomic_and_checksummed(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"sample_id":"one"}\n', encoding="utf-8")
    monkeypatch.setattr("agrinet.common.artifacts.repository_root", lambda: tmp_path)
    artifact = register_file_artifact(
        source, "datasets", "fixture-v1", "agrinet.sft.student/v1", "run-one", {"records": 1}
    )
    destination = tmp_path / artifact.path
    manifest = yaml.safe_load((destination.parent / "artifact.yaml").read_text(encoding="utf-8"))
    assert artifact.checksum == sha256_file(destination)
    assert manifest["metadata"]["records"] == 1
    assert not destination.with_suffix(destination.suffix + ".tmp").exists()
