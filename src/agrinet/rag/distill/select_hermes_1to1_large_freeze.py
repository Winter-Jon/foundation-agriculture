#!/usr/bin/env python3
"""Deterministically select the final balanced 560 Direct + 560 RAG candidate view."""
from __future__ import annotations

import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agrinet.data.rebuild_sft import CELLS, canonical_json_hash, cell_of, load_hermes_1to1_specification, long_direct_audit_errors, strict_rag_trajectory_errors, validate_hermes_1to1_rows
from agrinet.data.sft_recovery import read_jsonl, write_jsonl
from agrinet.rag.distill.collect_hermes_1to1_v2 import private_oracle_text_in_messages


def eligible_accepted_paths(root: Path) -> list[Path]:
    """Return only explicitly current collection artifacts.

    The checkpointed large run writes directories named
    ``large-<run-id>-<route>-<cell>`` rather than the old ``direct-*`` /
    ``rag-*`` convention.  Restrict discovery to those immutable large-run
    outputs and separately lineage-controlled Oracle recoveries; this prevents
    historical unreadable RAG collections from entering a new freeze merely
    because they have an ``accepted.jsonl`` file.  The language-aware Direct
    re-audit is a ``large-*`` directory and is included automatically.
    """
    paths: list[Path] = []
    for directory in sorted(root.iterdir() if root.is_dir() else []):
        if not directory.is_dir() or not (directory.name.startswith("large-") or directory.name.startswith("oracle-rag-")):
            continue
        path = directory / "accepted.jsonl"
        if path.is_file():
            paths.append(path)
    return paths


def accepted(root: Path, *, route: str) -> list[dict[str, Any]]:
    """Read current accepted rows for one route, based on provenance."""
    rows: list[dict[str, Any]] = []
    for path in eligible_accepted_paths(root):
        for row in read_jsonl(path):
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            generation_route = str(metadata.get("generation_route") or "")
            is_direct = generation_route == "direct_visual_comparison"
            is_rag = generation_route in {"blind_evidence", "oracle_grounded"}
            if (route == "direct" and is_direct) or (route == "rag" and is_rag):
                rows.append(row)
    return rows


def choose(rows: list[dict[str, Any]], *, route: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = load_hermes_1to1_specification(); selected: list[dict[str, Any]] = []; excluded: list[dict[str, Any]] = []
    if route == "direct":
        valid: list[dict[str, Any]] = []
        for row in rows:
            errors = long_direct_audit_errors(row, forbidden_hashes=set())
            if errors: excluded.append({"sample_id": row.get("sample_id"), "reason": sorted(set(errors))})
            else: valid.append(row)
        pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in valid: pairs[str((row.get("metadata") or {}).get("image_sha256") or "")].append(row)
        for question_type in ("open", "option"):
            for domain in ("disease", "pest"):
                choices = []
                for image, group in pairs.items():
                    cells = {cell_of(row) for row in group}; languages = {cell_of(row)[1] for row in group}
                    if len(group) == 2 and languages == {"en", "zh"} and cells == {(question_type, "en", domain), (question_type, "zh", domain)}:
                        choices.append(group)
                choices.sort(key=lambda group: str(group[0].get("sample_id") or ""))
                classes: Counter[str] = Counter(); letters: Counter[str] = Counter()
                for group in choices:
                    metadata = group[0].get("metadata") or {}; klass = str(metadata.get("canonical_class") or metadata.get("label_code") or ""); letter = str(metadata.get("correct_option") or "")
                    if not klass or classes[klass] >= int(spec["direct_max_per_class"]) or (question_type == "option" and letter in "ABCD" and letters[letter] >= int(spec["option_letter_floor"]) + 4):
                        excluded.extend({"sample_id": row.get("sample_id"), "reason": ["class_or_option_cap"]} for row in group); continue
                    if len([row for row in selected if cell_of(row) == (question_type, "en", domain)]) >= int(spec["direct_per_cell"]):
                        excluded.extend({"sample_id": row.get("sample_id"), "reason": ["cell_full"]} for row in group); continue
                    selected.extend(sorted(group, key=lambda row: cell_of(row)[1])); classes[klass] += 1; letters[letter] += 1
        return selected, excluded
    for cell in CELLS:
        pool = [row for row in rows if cell_of(row) == cell]
        valid = []
        for row in pool:
            errors = long_direct_audit_errors(row, forbidden_hashes=set()) if route == "direct" else strict_rag_trajectory_errors(row)
            if route == "rag" and private_oracle_text_in_messages(row.get("messages") or []): errors.append("private_oracle_text_leak")
            if errors:
                excluded.append({"sample_id": row.get("sample_id"), "reason": sorted(set(errors))}); continue
            valid.append(row)
        # Prefer Blind RAG to preserve its 56-row floor; stable order makes the view immutable.
        valid.sort(key=lambda row: (0 if route == "rag" and (row.get("metadata") or {}).get("generation_route") == "blind_evidence" else 1, str(row.get("sample_id") or "")))
        counts: Counter[str] = Counter(); letters: Counter[str] = Counter(); image_seen: set[str] = set()
        for row in valid:
            metadata = row.get("metadata") or {}; klass = str(metadata.get("canonical_class") or metadata.get("label_code") or "")
            image = str(metadata.get("image_sha256") or "")
            letter = str(metadata.get("correct_option") or "")
            if not klass or not image or image in image_seen or counts[klass] >= int(spec[f"{route}_max_per_class"]):
                excluded.append({"sample_id": row.get("sample_id"), "reason": ["class_or_image_cap"]}); continue
            # Preserve option balance while filling: no letter may be starved below floor.
            if cell[0] == "option" and letter in "ABCD" and letters[letter] >= int(spec["option_letter_floor"]) + 4:
                excluded.append({"sample_id": row.get("sample_id"), "reason": ["option_letter_overfill"]}); continue
            if len([item for item in selected if cell_of(item) == cell]) >= int(spec[f"{route}_per_cell"]):
                excluded.append({"sample_id": row.get("sample_id"), "reason": ["cell_full"]}); continue
            selected.append(row); counts[klass] += 1; image_seen.add(image); letters[letter] += 1
    return selected, excluded


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--collection-root", type=Path, required=True); parser.add_argument("--forbidden-hashes", type=Path, required=True); parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    direct, direct_excluded = choose(accepted(args.collection_root, route="direct"), route="direct")
    rag, rag_excluded = choose(accepted(args.collection_root, route="rag"), route="rag")
    forbidden = set(json.loads(args.forbidden_hashes.read_text(encoding="utf-8")))
    validation = validate_hermes_1to1_rows(direct, rag, forbidden_hashes=forbidden)
    report = {"schema_version": "agrinet.hermes-large-freeze-preflight/v1", "direct_candidates": len(direct), "rag_candidates": len(rag), "direct_sha256": canonical_json_hash(direct), "rag_sha256": canonical_json_hash(rag), "validation": validation, "direct_excluded": direct_excluded, "rag_excluded": rag_excluded}
    args.destination.mkdir(parents=True, exist_ok=True); write_jsonl(args.destination / "direct_selected.jsonl", direct); write_jsonl(args.destination / "rag_selected.jsonl", rag); (args.destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"direct": len(direct), "rag": len(rag), "valid": validation["valid"]}, ensure_ascii=False)); return 0 if validation["valid"] else 1


if __name__ == "__main__": raise SystemExit(main())
