"""Run E3.36 with R0/R1/R2 recovery and a 10% terminal-delivery ceiling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agrinet.rag import e328_artifact_audit as audit_base
from agrinet.rag import e328_classifier_campaign as base
from agrinet.rag.e336_classifier_full import (PROTOCOL,
    TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION)


def _configure() -> None:
    base.PROTOCOL = PROTOCOL; base.NEW_PROVENANCE = "new_e336"
    base.REUSED_PROVENANCE = "reused_e322_v3"; base.PROVENANCE_KEY = "e336_provenance"
    base.TRACE_BOUND_Q1 = True
    base.TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION = TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION
    base.OUTCOME_SCHEMA = "agrinet.e336-classifier-full-outcomes/v1"
    base.CONTINUATION_SCHEMA = "agrinet.e336-classifier-full-continuation/v1"
    base.REPORT_SCHEMA = "agrinet.e336-classifier-full-final-report/v1"
    audit_base.PROTOCOL = PROTOCOL; audit_base.NEW_PROVENANCE = "new_e336"
    audit_base.PROVENANCE_KEY = "e336_provenance"
    audit_base.TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION = TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION
    audit_base.AUDIT_SCHEMA = "agrinet.e336-classifier-full-artifact-audit/v1"
    audit_base.GATE_SCHEMA = "agrinet.e336-classifier-full-gate/v1"


def run_campaign(**kwargs):
    _configure()
    return base.run_campaign(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "source", "e322-campaign", "output-root", "private-registry"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--teacher-model", default="gpt-5.6-sol")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--authorize-live-collection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); _configure()
    if args.dry_run:
        print(json.dumps(base.validate_inputs(manifest=args.manifest, source=args.source, e322_campaign=args.e322_campaign), sort_keys=True)); return 0
    result = run_campaign(manifest=args.manifest, source=args.source, e322_campaign=args.e322_campaign, output_root=args.output_root, private_registry=args.private_registry, teacher_model=args.teacher_model, timeout=args.timeout, authorize_live_collection=args.authorize_live_collection)
    print(json.dumps({"classifier_gate_passed":result["classifier_gate_passed"],"rows":result["rows"]}, sort_keys=True))
    return 0 if result["classifier_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
