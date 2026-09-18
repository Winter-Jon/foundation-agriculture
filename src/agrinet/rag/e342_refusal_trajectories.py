"""Prepare immutable E3.42 refusal-trajectory inputs from E3.41 safe rejects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agrinet.rag.e322_presample import digest, rows
from agrinet.rag.e328_classifier_full import FLAGS

PROTOCOL = "agrinet.e342-refusal-trajectories-full/v1"
SCHEMA = "agrinet.e342-refusal-trajectories-manifest/v1"
ARTIFACT_ROOT_NAME = "e342-refusal-trajectories-full-v1"
CAMPAIGN_ID = "e342-refusal-trajectories-full-v1"
SHARD_SIZE = 64
EXPECTED_ROWS = 82
ROUND_FILES = ("r0.json", "q1.json", "r1.json", "q1-r1.json", "r2.json", "q1-r2.json")


def _write(path: Path, value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        raise ValueError(f"E3.42 immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def latest_outcomes(campaign_root: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for shard in sorted((campaign_root / "shards").glob("shard-*")):
        for name in ROUND_FILES:
            path = shard / "outcomes" / name
            if path.is_file():
                for outcome in json.loads(path.read_text()).get("outcomes") or []:
                    latest[outcome["sample_id"]] = outcome
    return latest


def prepare(*, e341_source: Path, e341_report: Path, e341_audit: Path, e341_gate: Path, e341_campaign: Path, output_root: Path) -> dict[str, Any]:
    root = Path(output_root)
    if root.name != ARTIFACT_ROOT_NAME:
        raise ValueError(f"{PROTOCOL} requires its dedicated fresh artifact root")
    targets = (root / "source.jsonl", root / "manifest.json", root / "prepare-audit.json")
    if any(path.exists() for path in targets):
        raise ValueError("E3.42 prepare outputs are immutable")
    source_rows = rows(e341_source)
    report = json.loads(e341_report.read_text())
    audit = json.loads(e341_audit.read_text())
    gate = json.loads(e341_gate.read_text())
    if report.get("protocol") != "agrinet.e341-visual-top3-rag-safe-subset-slots/v1":
        raise ValueError("E3.42 requires E3.41 slot-preserving report")
    if not audit.get("artifact_audit_passed") or not gate.get("partial_rag_input_gate_passed"):
        raise ValueError("E3.42 requires the E3.41 audited safe-terminal gate")
    if audit.get("source_sha256") != digest(e341_source):
        raise ValueError("E3.42 E3.41 audit/source SHA binding invalid")
    if gate.get("final_report_sha256") != digest(e341_report) or gate.get("artifact_audit_sha256") != digest(e341_audit):
        raise ValueError("E3.42 E3.41 gate SHA binding invalid")
    by = {row["sample_id"]: row for row in source_rows}
    queue = report.get("future_reject_queue") or []
    ids = [item.get("sample_id") for item in queue]
    if len(ids) != EXPECTED_ROWS or len(ids) != len(set(ids)) or any(not isinstance(sid, str) or sid not in by for sid in ids):
        raise ValueError("E3.42 future_reject queue binding invalid")
    latest = latest_outcomes(e341_campaign)
    if set(latest) != set(by):
        raise ValueError("E3.42 E3.41 terminal coverage invalid")
    frozen = []
    for sid in sorted(ids):
        parent = latest[sid]
        if parent.get("disposition") != "future_reject" or parent.get("delivery_status") != "delivered" or parent.get("quality") != "pass":
            raise ValueError(f"E3.42 parent is not a safe future_reject: {sid}")
        trajectory = Path(str(parent.get("trajectory_path") or ""))
        evidence = Path(str(parent.get("evidence_path") or ""))
        if not trajectory.is_file() or not evidence.is_file() or digest(trajectory) != parent.get("trajectory_sha256") or digest(evidence) != parent.get("evidence_sha256"):
            raise ValueError(f"E3.42 parent evidence binding invalid: {sid}")
        frozen.append({**by[sid], "e342_protocol": PROTOCOL, "e342_parent": {
            "parent_protocol": report["protocol"], "parent_outcome": {key: parent.get(key) for key in ("work_id", "request_id", "private_audit_request_id", "round", "attempt_ordinal", "quality_attempt_ordinal", "disposition", "semantic")},
            "trajectory_path": str(trajectory), "trajectory_sha256": digest(trajectory),
            "evidence_path": str(evidence), "evidence_sha256": digest(evidence),
        }, **FLAGS})
    root.mkdir(parents=True, exist_ok=True)
    targets[0].write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in frozen))
    bindings = {name: {"path": str(path), "sha256": digest(path)} for name, path in {
        "e341_source": e341_source, "e341_report": e341_report, "e341_audit": e341_audit, "e341_gate": e341_gate,
    }.items()}
    shards = []
    for index, start in enumerate(range(0, len(frozen), SHARD_SIZE)):
        batch = frozen[start:start + SHARD_SIZE]
        path = root / "manifests" / f"shard-{index:02d}.json"
        value = {"schema_version": "agrinet.e342-refusal-trajectories-shard-manifest/v1", "protocol": PROTOCOL, "shard_index": index, "rows": len(batch), "source_sha256": digest(targets[0]), "sample_ids": [row["sample_id"] for row in batch], "work_items": [{"work_id": f"R0:{row['sample_id']}:e342-refusal", "sample_id": row["sample_id"], "round": "R0", "attempt_ordinal": 0, "quality_attempt_ordinal": 0, "resume_operation": "refusal", "predecessor_request_id": None} for row in batch], "workers": 4, **FLAGS}
        _write(path, value)
        shards.append({"path": str(path), "sha256": digest(path), "rows": len(batch), "shard_index": index})
    controls = {"uncached_input_token_cap": 1_000_000, "global_micu_intent_limit": 8_000, "request_timeout_seconds": 180, "workers": 4, "reservation_uncached_tokens": {"refusal": 4_000, "private_refusal_audit": 2_000}, "recovery_rounds": ["R0", "R1", "R2"], "quality_repair_max": 1, "terminal_unknown_delivery_limit": 8}
    manifest = {"schema_version": SCHEMA, "protocol": PROTOCOL, "campaign_id": CAMPAIGN_ID, "source": str(targets[0]), "source_sha256": digest(targets[0]), "source_rows_expected": EXPECTED_ROWS, "immutable_inputs": bindings, "shards": shards, "workers_per_shard": 4, "orchestrator": "serial_immutable_shards", "shards_are_scheduling_only": True, "pipeline": ["parent_e341_public_evidence", "tool_free_refusal_generation", "deterministic_insufficient_evidence_renderer", "isolated_private_refusal_audit"], "forbidden_tools": ["agrinet_classifier_predict", "agrinet_classifier_expand", "agrinet_rag_search", "planner", "ranker", "agrinet_reject"], "collection_controls": controls, **FLAGS}
    _write(targets[1], manifest)
    summary = {"schema_version": "agrinet.e342-refusal-trajectories-prepare-audit/v1", "protocol": PROTOCOL, "source_sha256": digest(targets[0]), "manifest_sha256": digest(targets[1]), "rows": len(frozen), "shard_rows": [entry["rows"] for entry in shards], "parent_future_reject_rows": len(ids), "provider_intents_created": 0, **FLAGS}
    _write(targets[2], summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("e341-source", "e341-report", "e341-audit", "e341-gate", "e341-campaign", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    value = prepare(e341_source=args.e341_source, e341_report=args.e341_report, e341_audit=args.e341_audit, e341_gate=args.e341_gate, e341_campaign=args.e341_campaign, output_root=args.output_root)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
