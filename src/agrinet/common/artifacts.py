from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from agrinet.common.contracts import ArtifactRef, RunManifest
from agrinet.common.paths import repository_root
from agrinet.common.runtime import write_yaml_atomic


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_file_artifact(
    source: Path,
    artifact_type: str,
    artifact_id: str,
    schema_version: str,
    producer_run_id: str,
    metadata: dict[str, Any] | None = None,
) -> ArtifactRef:
    destination_dir = repository_root() / "outputs" / "artifacts" / artifact_type / artifact_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    artifact = ArtifactRef(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        path=destination.relative_to(repository_root()),
        producer_run_id=producer_run_id,
        schema_version=schema_version,
        checksum=sha256_file(destination),
        metadata=metadata or {},
    )
    write_yaml_atomic(destination_dir / "artifact.yaml", artifact.model_dump(mode="json"))
    return artifact


def attach_output_artifact(run_manifest: Path, artifact: ArtifactRef) -> None:
    """Attach an artifact to an existing run manifest without duplicating it."""
    import yaml

    manifest = RunManifest.model_validate(yaml.safe_load(run_manifest.read_text(encoding="utf-8")))
    outputs = [item for item in manifest.output_artifacts if item.artifact_id != artifact.artifact_id]
    manifest.output_artifacts = [*outputs, artifact]
    write_yaml_atomic(run_manifest, manifest.model_dump(mode="json"))
