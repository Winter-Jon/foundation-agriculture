from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.repository_audit import _approved_output_path, build_report


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_report_classifies_links_outputs_and_test_boundaries(tmp_path: Path) -> None:
    _write(tmp_path / "README.md", "[valid](docs/guide.md) [broken](missing.md)")
    _write(tmp_path / "docs/guide.md", "# Guide")
    _write(tmp_path / "docs/archive/history.md", "[missing](old-output.json)")
    _write(tmp_path / "tests/test_example.py", "def test_ok(): pass")
    _write(tmp_path / "tests/fixtures/input.json", "{}")
    _write(tmp_path / "tools/legacy/data-preparation/local_utility.py", "import shutil; shutil.move('datasets/source', 'outputs/archive')")
    _write(tmp_path / "outputs/runs/data/current/run-1/status.json", json.dumps({"status": "complete"}))
    _write(tmp_path / "outputs/runs/rag/legacy/logs/worker.log", "legacy")
    _write(tmp_path / "outputs/artifacts/item.txt", "artifact")
    _write(tmp_path / "outputs/vlm_sft/branch/v0/checkpoint-1/weights.bin", "weights")
    _write(tmp_path / "outputs/artifacts/datasets/dataset-v2/artifact.yaml", "artifact_id: dataset-v1\nstatus: planned\n")
    _write(
        tmp_path / "outputs/artifacts/datasets/agrinet-rag-recovery-pilot-v2/artifact.yaml",
        "artifact_id: agrinet-rag-recovery-pilot-v2\nseed: 2\nstatus: planned\n",
    )
    _write(
        tmp_path / "configs/experiments/data/data-sft-rag-recovery-pilot-v2.yaml",
        "id: data-sft-rag-recovery-pilot-v2\noutputs:\n  pilot_dir: outputs/artifacts/datasets/agrinet-rag-recovery-pilot-v2\nparameters:\n  seed: 2\n",
    )

    report = build_report(tmp_path)

    assert report["schema_version"] == "agrinet.repository-audit/v4"
    assert report["documentation"]["checked_relative_links"] == 3
    assert report["documentation"]["broken_by_scope"] == {"active": 1, "archive": 1}
    assert report["outputs"]["managed_runs"]["status_counts"] == {"complete": 1}
    assert report["outputs"]["managed_runs"]["raw_status_counts"] == {"complete": 1}
    assert report["outputs"]["legacy_run_layouts"] == ["outputs/runs/rag/legacy"]
    assert report["tests"]["pytest_modules"] == ["tests/test_example.py"]
    assert report["tests"]["legacy_utilities"] == [{"path": "tools/legacy/data-preparation/local_utility.py", "review_required": True}]
    assert report["migration_manifests"]["legacy_runs"] == [
        {
            "path": "outputs/runs/rag/legacy",
            "bytes": 6,
            "files": 1,
            "directories": 1,
            "legacy_markers": ["logs"],
            "references": [],
            "proposed_action": "archive_candidate_review",
            "approval_required": True,
        }
    ]
    utility = report["migration_manifests"]["legacy_utilities"][0]
    assert utility["mutation_signals"] == ["move"]
    assert utility["local_path_literals"] == ["datasets/source", "outputs/archive"]
    assert utility["proposed_action"] == "mutation_review_required"
    checkpoint = report["model_and_dataset_audit"]["checkpoints"][0]
    assert checkpoint["path"].endswith("outputs/vlm_sft/branch/v0/checkpoint-1")
    assert checkpoint["proposed_action"] == "retain_review"
    artifacts = {entry["artifact_id"]: entry for entry in report["model_and_dataset_audit"]["datasets"]}
    artifact = artifacts["dataset-v2"]
    assert artifact["identity_mismatch"]
    assert artifact["declared_status"] == "planned"
    recovery_artifact = artifacts["agrinet-rag-recovery-pilot-v2"]
    assert not recovery_artifact["identity_mismatch"]
    assert not recovery_artifact["config_identity_mismatch"]
    assert not recovery_artifact["seed_mismatch"]


def test_output_destination_is_limited_to_migration_directory(tmp_path: Path) -> None:
    allowed = _approved_output_path(tmp_path, Path("outputs/migration/audit/report.json"))
    assert allowed == tmp_path / "outputs/migration/audit/report.json"
    with pytest.raises(ValueError, match="outputs/migration"):
        _approved_output_path(tmp_path, Path("outputs/runs/report.json"))


def test_build_report_normalizes_historical_completed_status(tmp_path: Path) -> None:
    _write(tmp_path / "outputs/runs/vlm/fixture-v1/run/status.json", json.dumps({"status": "completed"}))
    _write(tmp_path / "outputs/runs/vlm/fixture-v1/failed/status.json", json.dumps({"status": "failed"}))

    report = build_report(tmp_path)

    managed = report["outputs"]["managed_runs"]
    assert managed["raw_status_counts"] == {"completed": 1, "failed": 1}
    assert managed["status_counts"] == {"complete": 1, "failed": 1}
    assert managed["historical_status_aliases"] == {"completed": "complete"}
    assert managed["failed_by_experiment"] == {"fixture-v1": 1}
