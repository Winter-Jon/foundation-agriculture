#!/usr/bin/env python3
"""Build a full-class HCV long-chain RAG plan without remote calls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agrinet.data.hcv_long_chain_sampling import build_plan, load_contract
from agrinet.data.io import DataError, read_jsonl, write_jsonl_atomic


def rows(path: Path) -> list[dict[str, Any]]:
    return [row for _, row in read_jsonl(path)]


def hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("image_sha256") or "") for row in rows(path) if str(row.get("image_sha256") or "")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--excluded-hashes", type=Path, nargs="*", default=())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-target", type=int, default=217)
    parser.add_argument("--seed", default="hcv-long-chain-v1")
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/sampling/hcv-long-chain-rag-contract-v1.yaml")
    args = parser.parse_args()
    try:
        load_contract(args.contract)
        excluded = set().union(*(hashes(path) for path in args.excluded_hashes)) if args.excluded_hashes else set()
        public, private, report = build_plan(rows(args.classes), rows(args.candidates), excluded, class_target=args.class_target, seed=args.seed)
    except DataError as exc:
        print(json.dumps({"ready": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    if args.output_dir.exists():
        print(json.dumps({"ready": False, "error": f"output directory exists: {args.output_dir}"}, ensure_ascii=False))
        return 1
    write_jsonl_atomic(args.output_dir / "public_teacher_plan.jsonl", public)
    write_jsonl_atomic(args.output_dir / "private_audit.jsonl", private)
    (args.output_dir / "plan_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ready": True, **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
