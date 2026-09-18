"""Prepare the E3.39 dynamic RAG safe subset from E3.38 terminals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agrinet.rag import e329_visual_top3_full as base
from agrinet.rag.e322_presample import digest

PROTOCOL = "agrinet.e339-visual-top3-rag-safe-subset/v1"


def _configure(protocol: str = PROTOCOL, schema: str = "agrinet.e339-visual-top3-rag-safe-subset-manifest/v1") -> None:
    base.PROTOCOL = protocol
    base.SCHEMA = schema


def prepare(*, protocol: str = PROTOCOL, schema: str = "agrinet.e339-visual-top3-rag-safe-subset-manifest/v1", required_root_name: str = "e339-visual-top3-rag-safe-subset-v1", **kwargs):
    root = Path(kwargs["output_root"])
    if required_root_name not in str(root):
        raise ValueError("E3.39 requires its dedicated immutable artifact root")
    report_path = Path(kwargs["e328_report"])
    audit_path = Path(kwargs["e328_audit"])
    gate_path = Path(kwargs["e328_gate"])
    report = json.loads(report_path.read_text())
    audit = json.loads(audit_path.read_text())
    gate = json.loads(gate_path.read_text())
    if report.get("protocol") != "agrinet.e338-classifier-full/v1":
        raise ValueError("E3.39 requires the E3.38 sample-level classifier report")
    if audit.get("protocol") != "agrinet.e338-classifier-full/v1":
        raise ValueError("E3.39 requires the matching E3.38 artifact audit")
    if gate.get("final_report_sha256") != digest(report_path):
        raise ValueError("E3.39 classifier gate/report SHA binding invalid")
    if gate.get("artifact_audit_sha256") != digest(audit_path):
        raise ValueError("E3.39 classifier gate/audit SHA binding invalid")
    if audit.get("source_sha256") != digest(Path(kwargs["e328_source"])):
        raise ValueError("E3.39 classifier audit/source SHA binding invalid")
    _configure(protocol, schema)
    return base.prepare(**kwargs, allow_partial_classifier=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("e328-source", "e328-report", "e328-audit", "e328-gate", "e327-source", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(e328_source=args.e328_source, e328_report=args.e328_report, e328_audit=args.e328_audit, e328_gate=args.e328_gate, e327_source=args.e327_source, output_root=args.output_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
