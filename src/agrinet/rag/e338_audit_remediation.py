"""Re-audit immutable E3.38 evidence after correcting prompt-text false positives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agrinet.rag import e328_artifact_audit as audit_base
from agrinet.rag.e338_classifier_campaign import _configure

AUDIT_NAME = "artifact-audit-remediation-v1.json"
GATE_NAME = "final-gate-decision-remediation-v1.json"


def run(*, source: Path, campaign_root: Path, e322_campaign: Path) -> dict:
    """Write a distinct audit/gate pair; never overwrite historical evidence."""
    _configure()
    audit_path = campaign_root / AUDIT_NAME
    gate_path = campaign_root / GATE_NAME
    audit_value = audit_base.audit(
        source=source, campaign_root=campaign_root, e322_campaign=e322_campaign, output=audit_path
    )
    report_path = campaign_root / "final-report.json"
    gate_value = audit_base.gate_decision(
        report=report_path, audit_report=audit_path, output=gate_path
    )
    return {"audit": audit_value, "gate": gate_value, "audit_path": str(audit_path), "gate_path": str(gate_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--e322-campaign", type=Path, required=True)
    args = parser.parse_args(argv)
    value = run(source=args.source, campaign_root=args.campaign_root, e322_campaign=args.e322_campaign)
    print(json.dumps({"artifact_audit_passed": value["audit"].get("artifact_audit_passed"), "partial_rag_input_gate_passed": value["gate"].get("partial_rag_input_gate_passed"), "audit_path": value["audit_path"], "gate_path": value["gate_path"]}, sort_keys=True))
    return 0 if value["gate"].get("partial_rag_input_gate_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
