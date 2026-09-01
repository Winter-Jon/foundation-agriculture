#!/usr/bin/env python3
"""Freeze a versioned *candidate* view after Hermes audit-rule updates.

This never mutates collection JSONL and does not claim final 560:560 validity.
It records rows that pass the current row contracts and separately accounts for
old Chinese Open mismatches that are now bilingual-answer equivalent but cannot
be promoted because their persisted rejection record lacks a tool response.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.data.rebuild_sft import canonical_json_hash
from agrinet.data.sft_recovery import read_jsonl, write_jsonl
from agrinet.rag.distill.collect_hermes_1to1_v2 import answer_matches_target
from agrinet.rag.distill.select_hermes_1to1_large_freeze import accepted, choose, eligible_accepted_paths


def cell(target: dict[str, Any]) -> str:
    return "/".join(str(target.get(key) or "") for key in ("question_type", "language", "task_domain"))


def recoverable_bilingual_mismatches(root: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: list[dict[str, str]] = []
    scanned = 0
    for path in sorted(root.glob("large-*-rag-open-zh-*/rejected.jsonl")):
        for record in read_jsonl(path):
            target = record.get("target") if isinstance(record.get("target"), dict) else {}
            generated = record.get("generated") if isinstance(record.get("generated"), dict) else {}
            if "answer_target_mismatch" not in (record.get("errors") or []) or not generated:
                continue
            scanned += 1
            final = str(generated.get("final") or "")
            if answer_matches_target(target, final):
                counts[cell(target)] += 1
                if len(examples) < 10:
                    examples.append({"sample_id": str(target.get("target_id") or ""), "cell": cell(target), "target_name": str(target.get("canonical_name") or "")})
    return {
        "scanned_old_chinese_open_mismatches": scanned,
        "now_bilingual_answer_equivalent": dict(sorted(counts.items())),
        "promotion_policy": "not_promoted: rejected records persist first/final text but no actual public tool response",
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise RuntimeError(f"candidate-pool destination already exists: {args.destination}")

    direct, direct_excluded = choose(accepted(args.collection_root, route="direct"), route="direct")
    rag, rag_excluded = choose(accepted(args.collection_root, route="rag"), route="rag")
    args.destination.mkdir(parents=True)
    write_jsonl(args.destination / "direct_current_contract.jsonl", direct)
    write_jsonl(args.destination / "rag_current_contract.jsonl", rag)
    report = {
        "schema_version": "agrinet.hermes-v2-candidate-pool/v1",
        "immutable_collection_sources": [str(path) for path in eligible_accepted_paths(args.collection_root)],
        "audit_contract": "current long Direct plus semantic Hermes RAG",
        "direct_rows": len(direct),
        "rag_rows": len(rag),
        "direct_sha256": canonical_json_hash(direct),
        "rag_sha256": canonical_json_hash(rag),
        "direct_excluded": direct_excluded,
        "rag_excluded": rag_excluded,
        "bilingual_answer_recovery_diagnostic": recoverable_bilingual_mismatches(args.collection_root),
        "formal_freeze_authorized": False,
    }
    (args.destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("direct_rows", "rag_rows", "direct_sha256", "rag_sha256")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
