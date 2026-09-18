"""Write an immutable E3.43 gate correction for the declared unknown-delivery allowance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agrinet.rag.e322_presample import digest
from agrinet.rag.e328_classifier_full import FLAGS
from agrinet.rag.e343_refusal_trajectories import PROTOCOL


def write_correction(*, report: Path, audit: Path, original_gate: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        return json.loads(output.read_text())
    value, audit_value, original = (json.loads(path.read_text()) for path in (report, audit, original_gate))
    limit = value.get("terminal_unknown_delivery_limit")
    unknown = value.get("terminal_unknown_delivery_count")
    complete = value.get("refusal_trajectory_complete_count")
    rows = value.get("rows")
    if value.get("protocol") != PROTOCOL or audit_value.get("protocol") != PROTOCOL or original.get("protocol") != PROTOCOL:
        raise ValueError("E3.43 correction protocol binding invalid")
    if not isinstance(limit, int) or not isinstance(unknown, int) or not isinstance(complete, int) or complete + unknown != rows:
        raise ValueError("E3.43 correction conservation invalid")
    passed = bool(audit_value.get("artifact_audit_passed") and not audit_value.get("errors") and unknown <= limit)
    corrected = {"schema_version": "agrinet.e343-refusal-trajectories-gate-correction/v1", "protocol": PROTOCOL,
                 "final_report_sha256": digest(report), "artifact_audit_sha256": digest(audit),
                 "superseded_gate_sha256": digest(original_gate), "rows": rows,
                 "refusal_trajectory_complete_count": complete, "terminal_unknown_delivery_count": unknown,
                 "terminal_unknown_delivery_limit": limit, "artifact_audit_passed": audit_value.get("artifact_audit_passed"),
                 "gate_rule": "zero non-delivery residuals and terminal_unknown_delivery_count <= declared limit",
                 "refusal_gate_passed": passed, "partial_refusal_gate_passed": passed,
                 "original_gate_refusal_gate_passed": original.get("refusal_gate_passed"),
                 "correction_reason": "original gate used an obsolete zero-unknown predicate despite the declared allowance",
                 "reject_executed": False, "next_action": "interview_user", **FLAGS}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(corrected, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return corrected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--original-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(write_correction(report=args.report, audit=args.audit, original_gate=args.original_gate, output=args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
