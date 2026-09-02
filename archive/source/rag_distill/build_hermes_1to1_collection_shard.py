#!/usr/bin/env python3
"""Derive one fresh v2 target per cell while excluding every contacted image."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.data.rebuild_sft import canonical_json_hash
from agrinet.data.sft_recovery import read_jsonl, write_jsonl


def valid_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() and path.stat().st_size else []


def contacted_hashes(root: Path) -> set[str]:
    hashes: set[str] = set()
    for path in root.glob("*/accepted.jsonl"):
        for row in valid_rows(path):
            hashes.add(str((row.get("metadata") or {}).get("image_sha256") or ""))
    for path in root.glob("*/rejected.jsonl"):
        for row in valid_rows(path):
            hashes.add(str((row.get("target") or {}).get("image_sha256") or ""))
    return hashes - {""}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-targets", type=Path, required=True)
    parser.add_argument("--rag-targets", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--per-cell", type=int, default=1)
    args = parser.parse_args()
    if args.per_cell < 1:
        raise ValueError("--per-cell must be positive")
    excluded = contacted_hashes(args.collection_root)
    direct_source, rag_source = read_jsonl(args.direct_targets), read_jsonl(args.rag_targets)
    direct: list[dict[str, Any]] = []
    direct_counts: Counter[tuple[str, str]] = Counter()
    selected_direct_hashes: set[str] = set()
    for row in direct_source:
        key = (str(row["question_type"]), str(row["task_domain"]))
        image_sha256 = str(row["image_sha256"])
        if direct_counts[key] >= args.per_cell or image_sha256 in excluded or image_sha256 in selected_direct_hashes:
            continue
        mates = [candidate for candidate in direct_source if candidate["image_sha256"] == image_sha256]
        if {candidate["language"] for candidate in mates} != {"en", "zh"}:
            continue
        direct.extend(sorted(mates, key=lambda candidate: candidate["language"]))
        direct_counts[key] += 1
        selected_direct_hashes.add(image_sha256)
    rag: list[dict[str, Any]] = []
    rag_counts: Counter[tuple[str, str, str]] = Counter()
    selected_rag_hashes: set[str] = set()
    for row in rag_source:
        cell = (str(row["question_type"]), str(row["language"]), str(row["task_domain"]))
        image_sha256 = str(row["image_sha256"])
        if rag_counts[cell] >= args.per_cell or image_sha256 in excluded or image_sha256 in selected_rag_hashes:
            continue
        rag.append(row); rag_counts[cell] += 1; selected_rag_hashes.add(image_sha256)
    report = {"schema_version":"agrinet.hermes-1to1-shard/v1", "per_cell":args.per_cell, "excluded_contact_hashes":len(excluded), "direct_rows":len(direct), "rag_rows":len(rag), "direct_sha256":canonical_json_hash(direct), "rag_sha256":canonical_json_hash(rag), "direct_cells":dict(Counter(f"{row['question_type']}/{row['language']}/{row['task_domain']}" for row in direct)), "rag_cells":dict(Counter(f"{row['question_type']}/{row['language']}/{row['task_domain']}" for row in rag)), "training_authorized":False}
    args.destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.destination / "direct_targets.jsonl", direct)
    write_jsonl(args.destination / "rag_targets.jsonl", rag)
    (args.destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if len(direct) == 8 * args.per_cell and len(rag) == 8 * args.per_cell else 1


if __name__ == "__main__":
    raise SystemExit(main())
