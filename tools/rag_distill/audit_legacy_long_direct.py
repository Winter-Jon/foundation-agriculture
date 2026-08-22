#!/usr/bin/env python3
"""Audit and deterministically select legacy long Direct comparisons intact."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agrinet.data.rebuild_sft import (
    CELLS, canonical_json_hash, cell_of, image_digest, long_direct_audit_errors,
    load_hermes_1to1_specification, query_image,
)
from agrinet.data.sft_recovery import read_jsonl, write_jsonl


def image_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("image_sha256") or query_image(row))


def class_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("canonical_class") or metadata.get("label_code") or "")


def enrich_legacy_rows(rows: list[dict[str, Any]], catalog_rows: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    """Attach auditable current mappings without touching legacy messages."""
    catalog = {str(row.get("query_image") or ""): row for row in catalog_rows}
    enriched: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = dict(source)
        image = query_image(row)
        match = catalog.get(image)
        messages = row.get("messages") or []
        user = next((str(message.get("content") or "") for message in messages if message.get("role") == "user"), "")
        language = "zh" if re.search(r"[\u4e00-\u9fff]", user) else "en"
        metadata = dict(row.get("metadata") or {})
        metadata.update({
            "source_dataset": "legacy_direct_2114",
            "source_row_index": index,
            "question_type": "open",
            "language": language,
            "paired_image_id": image,
            "uncertainty": load_hermes_1to1_specification()["uncertainty"],
        })
        try:
            metadata["image_sha256"] = image_digest(image, root)
        except Exception:
            metadata["image_sha256"] = ""
        if match:
            metadata.update({
                "canonical_class": str(match.get("final_label") or ""),
                "task_domain": str(match.get("task_domain") or ""),
                "source_sample_id": str(match.get("sample_id") or ""),
            })
        row["metadata"] = metadata
        row.setdefault("sample_id", f"legacy-direct-{index:04d}")
        enriched.append(row)
    return enriched


def select(rows: list[dict[str, Any]], *, forbidden_hashes: set[str]) -> dict[str, Any]:
    spec = load_hermes_1to1_specification()
    accepted, excluded = [], []
    for index, row in enumerate(rows):
        errors = long_direct_audit_errors(row, forbidden_hashes=forbidden_hashes, specification=spec)
        source_id = str(row.get("sample_id") or index)
        if errors:
            excluded.append({"source_row_index": index, "sample_id": source_id, "reason": ",".join(errors)})
        else:
            accepted.append((index, row))
    selected: list[dict[str, Any]] = []
    selected_images: set[str] = set()
    for question_type in ("open", "option"):
        for domain in ("disease", "pest"):
            pool = [row for _, row in accepted if cell_of(row)[0] == question_type and cell_of(row)[2] == domain]
            by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in pool: by_image[image_key(row)].append(row)
            pairs = []
            for image, siblings in by_image.items():
                langs = {str((row.get("metadata") or {}).get("language") or "") for row in siblings}
                if len(siblings) == 2 and langs == {"en", "zh"} and len({class_key(row) for row in siblings}) == 1:
                    pairs.append((image, siblings))
            pairs.sort(key=lambda item: (-sum(len(str((row.get("messages") or [{}])[-1].get("content") or "")) for row in item[1]), item[0]))
            classes: Counter[str] = Counter()
            for image, siblings in pairs:
                if len([row for row in selected if cell_of(row)[0] == question_type and cell_of(row)[2] == domain and cell_of(row)[1] == "en"]) >= spec["direct_per_cell"]:
                    break
                klass = class_key(siblings[0])
                if not klass or classes[klass] >= spec["direct_max_per_class"]: continue
                if image in selected_images: continue
                selected.extend(sorted(siblings, key=lambda row: cell_of(row)[1]))
                selected_images.add(image); classes[klass] += 1
    selected_ids = {id(row) for row in selected}
    for index, row in accepted:
        if id(row) not in selected_ids:
            excluded.append({"source_row_index": index, "sample_id": str(row.get("sample_id") or index), "reason": "selection_balance_or_pairing"})
    counts = Counter(cell_of(row) for row in selected)
    report = {
        "schema_version": "agrinet.legacy-direct-audit/v2",
        "source_rows": len(rows), "accepted_before_balancing": len(accepted),
        "selected_rows": len(selected), "selected_sha256": canonical_json_hash(selected),
        "excluded": excluded,
        "cells": {"/".join(cell): counts[cell] for cell in CELLS},
        "class_coverage": {"/".join(cell): len({class_key(row) for row in selected if cell_of(row) == cell}) for cell in CELLS},
    }
    return {"selected": selected, "report": report}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate-catalog", type=Path, required=True, help="Current class/image mapping catalog.")
    parser.add_argument("--forbidden-hashes", type=Path, required=True, help="JSON array of held-out/train-freeze image hashes")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    rows = enrich_legacy_rows(read_jsonl(args.source), read_jsonl(args.candidate_catalog), args.repo_root)
    result = select(rows, forbidden_hashes=set(json.loads(args.forbidden_hashes.read_text(encoding="utf-8"))))
    args.destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.destination / "audited_legacy_direct.jsonl", result["selected"])
    (args.destination / "audit.json").write_text(json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result["report"][key] for key in ("source_rows", "selected_rows", "selected_sha256")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
