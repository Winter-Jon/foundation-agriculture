#!/usr/bin/env python3
"""Re-audit historical accepted RAG rows for the long Hermes 1:1 contract."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.data.rebuild_sft import (
    CELLS, canonical_json_hash, cell_of, image_digest, load_hermes_1to1_specification,
    query_image, strict_rag_trajectory_errors,
)
from agrinet.data.sft_recovery import read_jsonl, write_jsonl


def enrich(row: dict[str, Any], *, source: str, root: Path) -> dict[str, Any]:
    result = dict(row)
    metadata = dict(result.get("metadata") or {})
    image = query_image(result)
    try:
        metadata["image_sha256"] = image_digest(image, root)
    except Exception:
        metadata["image_sha256"] = ""
    metadata["uncertainty"] = load_hermes_1to1_specification()["uncertainty"]
    metadata["historical_source"] = source
    result["metadata"] = metadata
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-list", type=Path, help="One accepted JSONL path per line.")
    source_group.add_argument("--source-root", type=Path, help="Recursively inspect accepted RAG files below this directory.")
    parser.add_argument("--forbidden-hashes", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    forbidden = set(json.loads(args.forbidden_hashes.read_text(encoding="utf-8")))
    paths = (
        [Path(line.strip()) for line in args.source_list.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
        if args.source_list else sorted(args.source_root.rglob("agent_sft.accepted.jsonl"))
    )
    accepted, excluded, seen_images = [], [], set()
    for path in paths:
        for index, raw in enumerate(read_jsonl(path)):
            row = enrich(raw, source=str(path), root=args.repo_root)
            sample_id = str(row.get("sample_id") or f"{path}:{index}")
            digest = str(row["metadata"]["image_sha256"])
            errors = strict_rag_trajectory_errors(row)
            if not digest: errors.append("missing_image_hash")
            if digest in forbidden: errors.append("image_isolation_overlap")
            if digest in seen_images: errors.append("duplicate_historical_image")
            if errors:
                excluded.append({"sample_id": sample_id, "source": str(path), "reason": ",".join(sorted(set(errors)))})
            else:
                seen_images.add(digest); accepted.append(row)
    spec = load_hermes_1to1_specification()
    selected: list[dict[str, Any]] = []
    for cell in CELLS:
        pool = [row for row in accepted if cell_of(row) == cell]
        pool.sort(key=lambda row: str(row.get("sample_id") or ""))
        classes: Counter[str] = Counter()
        letters: Counter[str] = Counter()
        for row in pool:
            metadata = row.get("metadata") or {}; klass = str(metadata.get("canonical_class") or metadata.get("label_code") or "")
            letter = str(metadata.get("correct_option") or "")
            if len([item for item in selected if cell_of(item) == cell]) >= spec["rag_per_cell"] or not klass or classes[klass] >= spec["rag_max_per_class"]:
                continue
            selected.append(row); classes[klass] += 1; letters[letter] += 1
    counts = Counter(cell_of(row) for row in selected)
    report = {
        "schema_version": "agrinet.hermes-1to1-rag-audit/v1",
        "source_files": [str(path) for path in paths], "source_rows": len(accepted) + len(excluded),
        "accepted_before_balancing": len(accepted), "selected_rows": len(selected),
        "selected_sha256": canonical_json_hash(selected), "excluded": excluded,
        "cells": {"/".join(cell): counts[cell] for cell in CELLS},
        "class_coverage": {"/".join(cell): len({str((row.get("metadata") or {}).get("canonical_class") or (row.get("metadata") or {}).get("label_code") or "") for row in selected if cell_of(row) == cell} - {""}) for cell in CELLS},
    }
    args.destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.destination / "audited_historical_rag.jsonl", selected)
    (args.destination / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("source_rows", "accepted_before_balancing", "selected_rows", "selected_sha256")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
