#!/usr/bin/env python3
"""Promote only fully persisted known rejections that pass the current contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agrinet.data.rebuild_sft import long_direct_audit_errors, strict_rag_trajectory_errors
from agrinet.data.sft_recovery import read_jsonl, write_jsonl
from archive.source.rag_distill.collect_hermes_1to1_v2 import answer_matches_target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--route", choices=("direct", "rag"), required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise RuntimeError(f"destination exists: {args.destination}")
    promoted, retained = [], []
    for record in read_jsonl(args.rejected):
        target = record.get("target") if isinstance(record.get("target"), dict) else {}
        row = record.get("trajectory") if isinstance(record.get("trajectory"), dict) else None
        if row is None:
            retained.append({**record, "re_audit_errors": ["missing_persisted_complete_trajectory"]})
            continue
        errors = long_direct_audit_errors(row, forbidden_hashes=set()) if args.route == "direct" else strict_rag_trajectory_errors(row)
        final = str((row.get("messages") or [{}])[-1].get("content") or "")
        if not answer_matches_target(target, final):
            errors.append("answer_target_mismatch")
        if errors:
            retained.append({**record, "re_audit_errors": sorted(set(errors))})
            continue
        metadata = dict(row.get("metadata") or {})
        row = {**row, "metadata": {**metadata, "re_audited_from": str(args.rejected), "re_audit_reason": "semantic_parser_and_bilingual_answer_repair"}}
        promoted.append(row)
    args.destination.mkdir(parents=True)
    write_jsonl(args.destination / "accepted.jsonl", promoted)
    write_jsonl(args.destination / "retained_rejected.jsonl", retained)
    (args.destination / "report.json").write_text(json.dumps({"source": str(args.rejected), "route": args.route, "promoted": len(promoted), "retained": len(retained)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print((args.destination / "report.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
