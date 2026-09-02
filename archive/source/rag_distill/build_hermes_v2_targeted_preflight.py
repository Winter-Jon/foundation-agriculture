#!/usr/bin/env python3
"""Extract the high-risk repair preflight from a fresh large-plan manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agrinet.data.rebuild_sft import canonical_json_hash
from agrinet.data.sft_recovery import read_jsonl, write_jsonl


def select(rows: list[dict], *, question_type: str, language: str, domain: str, count: int) -> list[dict]:
    chosen = [row for row in rows if (row.get("question_type"), row.get("language"), row.get("task_domain")) == (question_type, language, domain)][:count]
    if len(chosen) != count:
        raise ValueError(f"need {count} targets for {question_type}/{language}/{domain}, found {len(chosen)}")
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--per-cell", type=int, default=12)
    parser.add_argument("--offset", type=int, default=0, help="Skip already-contacted source-plan targets.")
    parser.add_argument("--rag-domains", nargs="+", choices=("disease", "pest"), default=("disease", "pest"))
    parser.add_argument("--no-direct", action="store_true", help="Extract only the requested Blind-RAG cells.")
    args = parser.parse_args()
    if args.destination.exists():
        raise RuntimeError(f"preflight destination exists: {args.destination}")
    direct_all = read_jsonl(args.plan / "direct_targets.jsonl")
    rag_all = read_jsonl(args.plan / "rag_targets.jsonl")
    # Direct needs the exact English/Chinese same-image pairs.
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    direct = [] if args.no_direct else [
        *select(direct_all, question_type="option", language="en", domain="pest", count=args.offset + args.per_cell)[args.offset:],
        *select(direct_all, question_type="option", language="zh", domain="pest", count=args.offset + args.per_cell)[args.offset:],
    ]
    # A source large plan can seed more than one immutable manifest.  Give
    # each derived request a stable, offset-qualified identity.
    direct = [{**row, "target_id": f"{row['target_id']}-offset{args.offset:03d}"} for row in direct]
    en_images = {row["image_sha256"] for row in direct if row["language"] == "en"}
    zh_images = {row["image_sha256"] for row in direct if row["language"] == "zh"}
    if direct and (en_images != zh_images or len(en_images) != args.per_cell):
        raise ValueError("Direct preflight lost bilingual image pairing")
    rag = [
        row
        for domain in args.rag_domains
        for row in select(rag_all, question_type="open", language="zh", domain=domain, count=args.offset + args.per_cell)[args.offset:]
    ]
    rag = [{**row, "target_id": f"{row['target_id']}-offset{args.offset:03d}"} for row in rag]
    if {row["image_sha256"] for row in direct} & {row["image_sha256"] for row in rag}:
        raise ValueError("Direct/RAG preflight image overlap")
    args.destination.mkdir(parents=True)
    if direct:
        write_jsonl(args.destination / "direct-option-en-pest.jsonl", [row for row in direct if row["language"] == "en"])
        write_jsonl(args.destination / "direct-option-zh-pest.jsonl", [row for row in direct if row["language"] == "zh"])
    for domain in args.rag_domains:
        write_jsonl(args.destination / f"rag-open-zh-{domain}.jsonl", [row for row in rag if row["task_domain"] == domain])
    report = {
        "schema_version": "agrinet.hermes-v2-targeted-preflight/v1",
        "source_plan": str(args.plan),
        "per_cell": args.per_cell, "source_offset": args.offset, "rag_domains": list(args.rag_domains),
        "direct_rows": len(direct), "rag_rows": len(rag),
        "direct_sha256": canonical_json_hash(direct), "rag_sha256": canonical_json_hash(rag),
        "checks": {"direct_bilingual_pairs": len(en_images), "direct_rag_image_overlap": 0},
        "purpose": "validate repaired Chinese/Open answer matching and semantic RAG trajectory before scaling",
        "training_authorized": False,
    }
    (args.destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
