#!/usr/bin/env python3
"""Build one immutable, balanced 32-RAG + 32-Direct Stage-A freeze.

The input view can contain surplus strict RAG rows.  This builder makes that
choice explicit and deterministic instead of silently training on the surplus.
It is deliberately separate from the historical 48-RAG mixed-freeze builder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agrinet.research.shared.catalog import evaluation_images, normalized
from agrinet.research.hcv.artifact_validation import validate_sft_row
from archive.source.rag_distill.validate_mixed_freeze import direct_errors

ROOT = Path(__file__).resolve().parents[3]
STANDARD_CELLS = tuple(
    f"standard/{question_type}/{language}/{domain}"
    for question_type in ("open", "option")
    for language in ("en", "zh")
    for domain in ("disease", "pest")
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def row_image(row: dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    value = meta.get("query_image") or (row.get("images") or [None])[0]
    return str(normalized(value) or "")


def row_cell(row: dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return "/".join(str(meta.get(key) or "") for key in ("trajectory_mode", "question_type", "language", "task_domain"))


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("sample_id") or ""), row_image(row))


def serialize_rag_for_training(row: dict[str, Any]) -> dict[str, Any]:
    """Retain only the query image required by the sole user placeholder.

    Retrieved reference evidence remains in the audited tool-response messages.
    Passing reference image paths without corresponding ``<image>`` placeholders
    violates Qwen3-VL's multimodal position-index contract.
    """
    item = dict(row)
    item["images"] = [row_image(row)]
    metadata = dict(item.get("metadata") or {})
    metadata["training_image_serialization"] = "query_image_only_matches_one_user_placeholder"
    item["metadata"] = metadata
    return item


def choose_rag(rows: list[dict[str, Any]], per_cell: int, surplus_offset: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row_cell(row)].append(row)
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for cell in STANDARD_CELLS:
        candidates = sorted(buckets[cell], key=row_key)
        if len(candidates) < per_cell:
            raise ValueError(f"RAG cell {cell} has {len(candidates)} rows; need {per_cell}")
        # An authorized nonzero offset rotates only cells that genuinely have
        # surplus.  It cannot alter sparse cells, quotas, or validators.
        offset = surplus_offset % len(candidates) if len(candidates) > per_cell else 0
        rotated = candidates[offset:] + candidates[:offset]
        chosen, surplus = rotated[:per_cell], rotated[per_cell:]
        selected.extend(chosen)
        coverage[cell] = {
            "available": len(candidates),
            "selection_offset": offset,
            "selected_sample_ids": [str(row.get("sample_id")) for row in chosen],
            "excluded_sample_ids": [str(row.get("sample_id")) for row in surplus],
        }
        excluded.extend({"cell": cell, "sample_id": row.get("sample_id"), "query_image": row_image(row), "reason": "deterministic_surplus_after_sample_id_sort"} for row in surplus)
    unexpected = sorted(set(buckets) - set(STANDARD_CELLS))
    if unexpected:
        raise ValueError(f"non-standard rows in validation input: {unexpected}")
    return selected, excluded, coverage


def validate_selected(rag: list[dict[str, Any]], direct: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rag:
        row_errors, _ = validate_sft_row(row)
        errors.extend(row_errors)
    errors.extend(error for index, row in enumerate(direct) for error in direct_errors(row, index))
    images = [row_image(row) for row in rag + direct]
    sample_ids = [str(row.get("sample_id") or "") for row in rag]
    if any(not image for image in images):
        errors.append("empty query image")
    if len(images) != len(set(images)):
        errors.append("duplicate query image in freeze")
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("duplicate RAG sample_id in freeze")
    for row in rag + direct:
        messages = row.get("messages")
        # The strict RAG and Direct validators independently require the
        # messages contract.  This conditional keeps the structural helper
        # usable with minimal synthetic rows in tests.
        if not isinstance(messages, list) or not messages:
            continue
        placeholders = sum(str(message.get("content") or "").count("<image>") for message in messages if isinstance(message, dict))
        if len(row.get("images") or []) != placeholders:
            errors.append(f"image placeholder mismatch for {row.get('sample_id')}")
    overlap = set(images) & evaluation_images()
    if overlap:
        errors.append(f"evaluation image overlap: {sorted(overlap)[:5]}")
    return errors


def build(rag_path: Path, direct_path: Path, destination: Path, per_cell: int = 4, surplus_offset: int = 0) -> dict[str, Any]:
    rag_path = rag_path.resolve()
    direct_path = direct_path.resolve()
    destination = destination.resolve()
    rag_input, direct = read_jsonl(rag_path), read_jsonl(direct_path)
    if len(direct) != 32:
        raise ValueError(f"Direct input must have exactly 32 rows, got {len(direct)}")
    rag, excluded, coverage = choose_rag(rag_input, per_cell, surplus_offset)
    rag = [serialize_rag_for_training(row) for row in rag]
    if len(rag) != len(STANDARD_CELLS) * per_cell:
        raise ValueError(f"selected RAG has {len(rag)} rows, expected 32")
    errors = validate_selected(rag, direct)
    if errors:
        raise ValueError("freeze validation failed: " + "; ".join(errors[:10]))
    data_rows = rag + direct
    serialized = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in data_rows).encode()
    data_hash = sha256_bytes(serialized)
    manifest = {
        "schema_version": "agrinet.rag-sft-validation-freeze/v1",
        "artifact_id": destination.name,
        "data_sha256": data_hash,
        "rows": 64, "rag_rows": 32, "direct_rows": 32,
        "selection_policy": "four rows per standard cell, deterministic sample_id then query-image sort with authorized surplus rotation",
        "surplus_offset": surplus_offset,
        "source_rag": str(rag_path.relative_to(ROOT)),
        "source_direct": str(direct_path.relative_to(ROOT)),
        "source_rag_sha256": sha256_bytes(rag_path.read_bytes()),
        "source_direct_sha256": sha256_bytes(direct_path.read_bytes()),
        "immutable": True, "sft_runs_allowed_for_hash": 1,
    }
    if destination.exists():
        old = destination / "manifest.json"
        if not old.exists() or json.loads(old.read_text(encoding="utf-8")).get("data_sha256") != data_hash:
            raise FileExistsError(f"immutable freeze destination conflicts: {destination}")
        return {"artifact": str(destination), "data_sha256": data_hash, "reused_existing_immutable_freeze": True}
    destination.mkdir(parents=True)
    write_jsonl(destination / "data.jsonl", data_rows)
    write_jsonl(destination / "rag.jsonl", rag)
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "valid": True, "data_sha256": data_hash, "rag_input_rows": len(rag_input),
        "rag_selected_rows": len(rag), "direct_selected_rows": len(direct),
        "unique_query_images": len({row_image(row) for row in data_rows}),
        "rag_cell_coverage": coverage, "excluded_rag_surplus": excluded,
        "direct_coverage": dict(sorted(Counter("/".join(str((row.get("metadata") or {}).get(key)) for key in ("language", "task_domain", "question_type")) for row in direct).items())),
        "validation_errors": [],
    }
    (destination / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Retain a compact generic summary so existing read-only mixed-freeze
    # validators can inspect this new artifact without assuming 48 RAG rows.
    statistics = {
        "rows": 64, "rag_rows": 32, "direct_rows": 32,
        "unique_images": report["unique_query_images"],
        "rag_coverage": {cell: len(details["selected_sample_ids"]) for cell, details in coverage.items()},
        "direct_coverage": report["direct_coverage"],
        "valid": True,
    }
    (destination / "statistics.json").write_text(json.dumps(statistics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"artifact": str(destination), "data_sha256": data_hash, "reused_existing_immutable_freeze": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-file", type=Path, required=True)
    parser.add_argument("--direct-file", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--surplus-offset", type=int, default=0, help="Authorized deterministic rotation for cells with surplus rows only.")
    args = parser.parse_args()
    print(json.dumps(build(args.rag_file, args.direct_file, args.destination, surplus_offset=args.surplus_offset), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
