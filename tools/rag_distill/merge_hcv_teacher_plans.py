#!/usr/bin/env python3
"""Merge disjoint public HCV plans and their private audit sidecars."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, nargs="+", required=True)
    p.add_argument("--private-audit", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-rows", type=int, default=64)
    p.add_argument("--per-cell", type=int, default=8)
    p.add_argument("--cell-count", type=int, default=8)
    args = p.parse_args()
    if len(args.plan) != len(args.private_audit) or args.output_dir.exists():
        raise SystemExit("plan/private inputs must pair and output must not exist")
    plans = [row for path in args.plan for row in read(path)]
    audits = [row for path in args.private_audit for row in read(path)]
    ids = [str(row.get("sample_id") or "") for row in plans]
    hashes = [str(row.get("image_sha256") or "") for row in plans]
    audit_ids = [str(row.get("sample_id") or "") for row in audits]
    if not ids or len(ids) != args.expected_rows or len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)) or not all(hashes):
        raise SystemExit(f"merged public plan must have {args.expected_rows} unique nonempty image hashes and IDs")
    if set(ids) != set(audit_ids) or len(audit_ids) != len(set(audit_ids)):
        raise SystemExit("private audit must exactly cover merged public plan")
    if any({"final_label", "final_label_zh", "audit_truth_code"} & set(row) for row in plans):
        raise SystemExit("public plan contains private truth fields")
    counts = {}
    for row in plans:
        cell = "/".join(str(row.get(k) or "") for k in ("question_type", "language", "task_domain"))
        counts[cell] = counts.get(cell, 0) + 1
    if set(counts.values()) != {args.per_cell} or len(counts) != args.cell_count:
        raise SystemExit(f"merged plan is not {args.per_cell}-per-cell: {counts}")
    args.output_dir.mkdir(parents=True)
    write(args.output_dir / "teacher_plan.jsonl", plans)
    write(args.output_dir / "private_audit.jsonl", audits)
    report = {"schema_version": "agrinet.hcv-teacher-plan-merge/v1", "rows": args.expected_rows, "counts": dict(sorted(counts.items())), "public_truth_free": True, "unique_image_hashes": True}
    (args.output_dir / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
