#!/usr/bin/env python3
"""Merge disjoint accepted HCV collection shards into an auditable full set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, nargs="+", required=True)
    parser.add_argument("--private-audit", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--exclude-sample-id", type=Path, nargs="*", default=())
    parser.add_argument("--private-audit-output", type=Path)
    args = parser.parse_args()
    rows = [row for path in args.accepted for row in read_jsonl(path)]
    ids = [str(row.get("sample_id") or "") for row in rows]
    excluded_ids = {
        str(row.get("sample_id") or "")
        for path in args.exclude_sample_id
        for row in read_jsonl(path)
    }
    expected = {
        str(row.get("sample_id") or "")
        for path in args.private_audit
        for row in read_jsonl(path)
        if str(row.get("sample_id") or "") not in excluded_ids
    }
    if args.private_audit_output:
        private_rows = [
            row
            for path in args.private_audit
            for row in read_jsonl(path)
            if str(row.get("sample_id") or "") not in excluded_ids
        ]
        if {str(row.get("sample_id") or "") for row in private_rows} != expected:
            raise SystemExit("filtered private audit output does not match expected coverage")
        write_jsonl(args.private_audit_output, private_rows)
    observed = set(ids)
    report = {
        "schema_version": "agrinet.hcv-collection-merge/v1",
        "sources": [str(path) for path in args.accepted],
        "private_audit": [str(path) for path in args.private_audit],
        "excluded_sample_ids": sorted(excluded_ids),
        "rows": len(rows),
        "expected_rows": len(expected),
        "invariants": {
            "nonempty_ids": all(ids),
            "unique_ids": len(ids) == len(observed),
            "exact_private_audit_coverage": observed == expected,
        },
    }
    report["merge_authorized"] = all(report["invariants"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["merge_authorized"]:
        raise SystemExit(f"HCV merge gate failed: {report}")
    write_jsonl(args.output, rows)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
