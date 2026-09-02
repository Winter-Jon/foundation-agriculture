#!/usr/bin/env python3
"""Materialize one bounded Blind collection shard per audit-deficit cell."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from agrinet.data.sft_recovery import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    shortages = json.loads(args.audit_report.read_text(encoding="utf-8"))["selection"]["shortages"]
    cells: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(args.eligible):
        cells["/".join(str(row[key]) for key in ("question_type", "language", "task_domain"))].append(row)
    manifest = {"schema_version": "agrinet.reconstructive-collection-shards/v1", "source": str(args.eligible), "shards": {}, "training_authorized": False}
    for cell, deficit in shortages.items():
        if not deficit["rows"]:
            continue
        rows = cells[cell]
        path = args.destination / cell.replace("/", "-") / "blind_targets.jsonl"
        write_jsonl(path, rows)
        manifest["shards"][cell] = {"plan_file": str(path), "eligible_targets": len(rows), "stop_after_accepted": deficit["rows"], "class_shortage": deficit["classes"]}
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
