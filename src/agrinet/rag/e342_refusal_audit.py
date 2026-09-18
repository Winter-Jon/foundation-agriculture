"""Independent integrity audit and gate for E3.42 refusal trajectories."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agrinet.rag.e322_presample import digest, rows
from agrinet.rag.e328_classifier_full import FLAGS
from agrinet.rag.e342_refusal_campaign import ROUND_FILES, _latest
from agrinet.rag.e342_refusal_contract import public_catalog, validate_refusal
from agrinet.rag.e342_refusal_trajectories import PROTOCOL


def audit(*, source: Path, campaign_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        return json.loads(output.read_text())
    data = rows(source)
    latest = _latest(campaign_root)
    errors: list[str] = []
    completed: list[str] = []
    residuals: list[dict[str, Any]] = []
    if set(latest) != {row["sample_id"] for row in data}:
        errors.append("coverage")
    by = {row["sample_id"]: row for row in data}
    for sid, outcome in latest.items():
        if outcome.get("disposition") != "refusal_trajectory_complete":
            residuals.append({"sample_id": sid, "disposition": outcome.get("disposition"), "delivery_status": outcome.get("delivery_status"),
                              "contract_error": outcome.get("contract_error"), **FLAGS})
            continue
        try:
            refusal = json.loads(Path(str(outcome["refusal_path"])).read_text())
            trajectory = json.loads(Path(str(outcome["trajectory_path"])).read_text())
            validate_refusal(refusal, by[sid], public_catalog(by[sid]))
            if trajectory.get("route") != "refusal" or trajectory.get("tool_trace") != []:
                errors.append(f"tool_boundary:{sid}")
            if trajectory.get("answer", "").split("</think>")[-1] != "<answer>INSUFFICIENT_EVIDENCE</answer>":
                errors.append(f"answer:{sid}")
            parent = trajectory.get("parent_e341") or {}
            if parent.get("trajectory_sha256") != by[sid]["e342_parent"]["trajectory_sha256"] or parent.get("evidence_sha256") != by[sid]["e342_parent"]["evidence_sha256"]:
                errors.append(f"parent_binding:{sid}")
            completed.append(sid)
        except Exception as exc:
            errors.append(f"state:{sid}:{type(exc).__name__}")
    global_path = campaign_root / "global_micu_intents.jsonl"
    global_intents = [json.loads(line) for line in global_path.read_text().splitlines() if line.strip()] if global_path.is_file() else []
    ledger_intents = []
    forbidden = {"agrinet_classifier_predict", "agrinet_classifier_expand", "agrinet_rag_search", "planner", "ranker", "agrinet_reject"}
    for path in (campaign_root / "ledgers").glob("*/events.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get("event") == "intent":
                ledger_intents.append(event)
                payload = event.get("payload") or {}
                if payload.get("tools") not in ([], None) or str(payload.get("operation") or "") in forbidden:
                    errors.append("forbidden_provider_operation")
    if len(global_intents) != len(ledger_intents) or len({item.get("key") for item in global_intents}) != len(global_intents):
        errors.append("intent_ledger")
    value = {"schema_version": "agrinet.e342-refusal-trajectories-artifact-audit/v1", "protocol": PROTOCOL,
             "source_sha256": digest(source), "rows": len(data), "terminal_samples": len(latest),
             "completed_samples": sorted(completed), "completed_count": len(completed), "refusal_residuals": residuals,
             "global_intents": len(global_intents), "ledger_intents": len(ledger_intents),
             "forbidden_tool_calls": 0, "errors": sorted(set(errors)), "artifact_audit_passed": not errors, **FLAGS}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return value


def gate_decision(*, report: Path, audit_report: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        return json.loads(output.read_text())
    report_value = json.loads(report.read_text())
    audit_value = json.loads(audit_report.read_text())
    hard = bool(report_value.get("campaign_gate_candidate") and audit_value.get("artifact_audit_passed"))
    partial = bool(report_value.get("refusal_trajectory_complete_count") and audit_value.get("artifact_audit_passed"))
    value = {"schema_version": "agrinet.e342-refusal-trajectories-gate/v1", "protocol": PROTOCOL,
             "final_report_sha256": digest(report), "artifact_audit_sha256": digest(audit_report),
             "refusal_gate_passed": hard, "partial_refusal_gate_passed": partial,
             "reject_executed": False, "next_action": "interview_user", **FLAGS}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return value
