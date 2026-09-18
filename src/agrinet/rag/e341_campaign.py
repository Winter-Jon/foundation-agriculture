"""Execute clean E3.41 slot-preserving fixed-query visual Top-3 RAG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agrinet.rag import e329_artifact_audit as audit
from agrinet.rag import e329_campaign as base
from agrinet.rag import e329_visual_top3_full as source
from agrinet.rag.e341_visual_top3_safe_subset_slots import PROTOCOL, SCHEMA


def _configure() -> None:
    source.PROTOCOL = PROTOCOL
    source.SCHEMA = SCHEMA
    base.PROTOCOL = PROTOCOL
    audit.PROTOCOL = PROTOCOL


def run_campaign(**kwargs):
    _configure()
    return base.run_campaign(**kwargs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "source", "output-root", "private-registry"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rag-endpoint", required=True)
    parser.add_argument("--teacher-model", default="gpt-5.6-sol")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--authorize-live-collection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    _configure()
    if args.dry_run:
        print(json.dumps(base.validate_inputs(manifest=args.manifest, source=args.source, endpoint=args.rag_endpoint), sort_keys=True))
        return 0
    value = run_campaign(manifest=args.manifest, source=args.source, output_root=args.output_root, private_registry=args.private_registry, rag_endpoint=args.rag_endpoint, teacher_model=args.teacher_model, timeout=args.timeout, authorize_live_collection=args.authorize_live_collection)
    print(json.dumps({"rag_gate_passed": value["rag_gate_passed"], "rows": value["rows"]}, sort_keys=True))
    return 0 if value["rag_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
