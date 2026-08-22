#!/usr/bin/env python3
"""Report formal Hermes v2 cell shortfalls from a derived candidate pool."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agrinet.data.rebuild_sft import CELLS, canonical_json_hash, load_hermes_1to1_specification
from agrinet.data.sft_recovery import read_jsonl


def cell_of(row: dict) -> tuple[str, str, str]:
    metadata = row.get("metadata") or {}
    return tuple(str(metadata.get(key) or "") for key in ("question_type", "language", "task_domain"))  # type: ignore[return-value]


def report_route(rows: list[dict], route: str, spec: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for cell in CELLS:
        members = [row for row in rows if cell_of(row) == cell]
        classes = Counter(str((row.get("metadata") or {}).get("canonical_class") or "") for row in members)
        classes.pop("", None)
        letters = Counter(str((row.get("metadata") or {}).get("correct_option") or "") for row in members)
        target_rows = int(spec[f"{route}_per_cell"])
        target_classes = int(spec[f"{route}_min_classes"])
        entry = {
            "rows": len(members),
            "row_shortfall": max(0, target_rows - len(members)),
            "classes": len(classes),
            "class_shortfall": max(0, target_classes - len(classes)),
            "max_class_count": max(classes.values(), default=0),
        }
        if cell[0] == "option":
            entry["option_letters"] = {letter: letters[letter] for letter in "ABCD"}
            entry["option_letter_shortfall"] = {letter: max(0, int(spec["option_letter_floor"]) - letters[letter]) for letter in "ABCD"}
        result["/".join(cell)] = entry
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    direct = read_jsonl(args.candidate_pool / "direct_current_contract.jsonl")
    rag = read_jsonl(args.candidate_pool / "rag_current_contract.jsonl")
    spec = load_hermes_1to1_specification()
    payload = {
        "schema_version": "agrinet.hermes-v2-shortage-report/v1",
        "candidate_pool": str(args.candidate_pool),
        "candidate_hashes": {"direct": canonical_json_hash(direct), "rag": canonical_json_hash(rag)},
        "formal_target": {"direct_rows": 560, "rag_rows": 560, "per_cell": 70, "min_classes": 35},
        "direct": report_route(direct, "direct", spec),
        "rag": report_route(rag, "rag", spec),
        "formal_freeze_authorized": False,
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"destination": str(args.destination), "direct_rows": len(direct), "rag_rows": len(rag)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
