#!/usr/bin/env python3
"""Record interrupted large-plan starts and emit safe continuation/resend manifests."""
from __future__ import annotations

import argparse, json
from pathlib import Path

from agrinet.data.sft_recovery import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--unknown-first", action="store_true")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists(): raise RuntimeError(f"destination exists: {args.destination}")
    rows = read_jsonl(args.targets)
    if not rows: raise ValueError("empty target list")
    args.destination.mkdir(parents=True)
    unknown, remaining = (rows[:1], rows[1:]) if args.unknown_first else ([], rows)
    if unknown:
        write_jsonl(args.destination / "unknown_delivery.jsonl", [{"target": row, "errors": ["unknown_delivery_or_runtime_error"], "error": "large_queue_cancelled_while_first_teacher_request_had_no_checkpoint", "resend_eligible": True, "trajectory_number": 1} for row in unknown])
        write_jsonl(args.destination / "resend_targets.jsonl", [{**row, "resend_of": {"reason": "unknown_delivery_or_runtime_error", "trajectory_number": 2}} for row in unknown])
    write_jsonl(args.destination / "continuation_targets.jsonl", remaining)
    (args.destination / "report.json").write_text(json.dumps({"source": str(args.targets), "unknown": len(unknown), "resend": len(unknown), "continuation": len(remaining)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print((args.destination / "report.json").read_text(encoding="utf-8"))


if __name__ == "__main__": main()
