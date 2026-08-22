#!/usr/bin/env python3
"""Derive an auditable, deficit-targeted large-007 collection manifest.

This preserves the immutable large plan and intentionally withholds the
high-risk Blind open/zh/pest cell: its Oracle capacity is exhausted and its
new-image Blind preflight must be improved before a scale run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agrinet.data.rebuild_sft import canonical_json_hash
from agrinet.data.sft_recovery import read_jsonl, write_jsonl


DIRECT_IMAGES = {
    ("open", "disease"): 12,
    ("open", "pest"): 70,
    ("option", "disease"): 60,
    ("option", "pest"): 12,
}
RAG_CELLS = (
    ("open", "en", "disease"), ("open", "en", "pest"),
    ("open", "zh", "disease"),
    ("option", "en", "disease"), ("option", "en", "pest"),
    ("option", "zh", "disease"), ("option", "zh", "pest"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise RuntimeError(f"destination exists: {args.destination}")

    direct_all = read_jsonl(args.plan / "direct_targets.jsonl")
    rag_all = read_jsonl(args.plan / "rag_targets.jsonl")
    direct: list[dict] = []
    for (question_type, domain), image_count in DIRECT_IMAGES.items():
        selected = [row for row in direct_all if row.get("question_type") == question_type and row.get("task_domain") == domain]
        by_language = {language: [row for row in selected if row.get("language") == language][:image_count] for language in ("en", "zh")}
        if len(by_language["en"]) != image_count or len(by_language["zh"]) != image_count:
            raise ValueError(f"insufficient Direct plan capacity: {question_type}/{domain}")
        if [row["image_sha256"] for row in by_language["en"]] != [row["image_sha256"] for row in by_language["zh"]]:
            raise ValueError(f"lost bilingual pairing: {question_type}/{domain}")
        direct.extend(by_language["en"] + by_language["zh"])

    rag: list[dict] = []
    for cell in RAG_CELLS:
        selected = [row for row in rag_all if (row.get("question_type"), row.get("language"), row.get("task_domain")) == cell][:140]
        if len(selected) != 140:
            raise ValueError(f"insufficient RAG plan capacity: {'/'.join(cell)}")
        rag.extend(selected)
    direct_hashes = {row["image_sha256"] for row in direct}
    rag_hashes = {row["image_sha256"] for row in rag}
    if direct_hashes & rag_hashes:
        raise ValueError("Direct/RAG supplement image overlap")
    args.destination.mkdir(parents=True)
    for row in direct:
        key = f"direct-{row['question_type']}-{row['language']}-{row['task_domain']}"
        path = args.destination / f"{key}.jsonl"
        existing = read_jsonl(path) if path.exists() else []
        write_jsonl(path, existing + [row])
    for row in rag:
        key = f"rag-{row['question_type']}-{row['language']}-{row['task_domain']}"
        path = args.destination / f"{key}.jsonl"
        existing = read_jsonl(path) if path.exists() else []
        write_jsonl(path, existing + [row])
    report = {
        "schema_version": "agrinet.hermes-v2-large007-targeted-supplement/v1",
        "source_plan": str(args.plan), "direct_rows": len(direct), "rag_rows": len(rag),
        "direct_unique_images": len(direct_hashes), "rag_unique_images": len(rag_hashes),
        "direct_sha256": canonical_json_hash(direct), "rag_sha256": canonical_json_hash(rag),
        "excluded_high_risk_cell": "open/zh/pest",
        "excluded_reason": "Blind preflight is not yet scale-authorized; Oracle capacity is 14/14",
        "training_authorized": False,
    }
    (args.destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
