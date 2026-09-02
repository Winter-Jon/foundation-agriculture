#!/usr/bin/env python3
"""Build the immutable internal Stage-A 192-row RAG diagnostic.

This is deliberately a *selection* tool: it does not call a model or Milvus.
It selects 24 held-out rows from each public language/domain/answer-type cell,
checks that none shares an image or source sample with the supplied SFT freeze,
and writes a content hash plus selection report.  The source is the retained
618-row manifest, but this tool never schedules that formal evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


CELLS = [(language, domain, question_type)
         for question_type in ("open", "option")
         for language in ("en", "zh")
         for domain in ("disease", "pest")]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"{path}: no JSONL files")
    for file in files:
        for line_no, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{file}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{file}:{line_no}: expected object")
            rows.append(row)
    return rows


def row_key(row: dict[str, Any], seed: str) -> str:
    # Hash ordering is stable even if the upstream manifest's shard ordering changes.
    return hashlib.sha256(f"{seed}:{row.get('id')}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", default="stagea-v3-matched-192-v1")
    parser.add_argument("--per-cell", type=int, default=24)
    args = parser.parse_args()

    if args.per_cell < 1:
        raise SystemExit("--per-cell must be positive")
    source = load_jsonl(args.source)
    training = load_jsonl(args.freeze)
    source_ids = [str(row.get("id") or "") for row in source]
    if len(source_ids) != len(set(source_ids)) or not all(source_ids):
        raise SystemExit("source manifest must have non-empty unique ids")

    training_images = {str(image) for row in training for image in row.get("images") or []}
    training_ids = {str(row.get("sample_id") or "") for row in training}
    selected: list[dict[str, Any]] = []
    cell_report: dict[str, dict[str, Any]] = {}
    for language, domain, question_type in CELLS:
        key = f"{language}/{domain}/{question_type}"
        pool = [row for row in source if (row.get("language"), row.get("task_domain"), row.get("question_type")) == (language, domain, question_type)]
        eligible = [row for row in pool if str(row.get("image_path") or "") not in training_images and str(row.get("source_sample_id") or "") not in training_ids]
        if len(eligible) < args.per_cell:
            raise SystemExit(f"{key}: only {len(eligible)} isolated rows, need {args.per_cell}")
        chosen = sorted(eligible, key=lambda row: row_key(row, args.seed))[:args.per_cell]
        selected.extend(chosen)
        cell_report[key] = {
            "source_rows": len(pool), "isolated_rows": len(eligible),
            "selected_rows": len(chosen),
            "known_bucket_counts": dict(sorted(Counter(str(row.get("known_bucket") or "unknown") for row in chosen).items())),
        }

    if len(selected) != len(CELLS) * args.per_cell:
        raise SystemExit("unexpected diagnostic row count")
    encoded = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded, encoding="utf-8")
    report = {
        "schema_version": "agrinet.rag-sft-stagea-matched-diagnostic/v1",
        "purpose": "internal_192_row_matched_diagnostic_not_formal_618",
        "source": str(args.source), "freeze": str(args.freeze),
        "seed": args.seed, "per_cell": args.per_cell, "rows": len(selected),
        "sha256": digest, "training_image_overlap_count": 0,
        "training_sample_overlap_count": 0, "cells": cell_report,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(selected), "sha256": digest, "report": str(args.report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
