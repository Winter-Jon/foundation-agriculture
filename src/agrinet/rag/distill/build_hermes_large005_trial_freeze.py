#!/usr/bin/env python3
"""Create a clearly noncanonical Hermes SFT smoke freeze from large-005.

This deliberately does *not* validate the final 560:560 collection contract.
It exists only to prove that conversion and local Hermes SFT launch work with
newly collected, strict-row-audited data.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from agrinet.data.rebuild_sft import (
    canonical_json_hash,
    long_direct_audit_errors,
    strict_rag_trajectory_errors,
)
from agrinet.data.sft_recovery import read_jsonl, write_jsonl


def answer_matches_target(row: dict) -> bool:
    metadata = row.get("metadata") or {}
    content = str((row.get("messages") or [{}])[-1].get("content") or "")
    answer = content.split("<answer>", 1)[-1].split("</answer>", 1)[0].strip()
    return (
        answer == str(metadata.get("correct_option") or "")
        if metadata.get("question_type") == "option"
        else answer.casefold() == str(metadata.get("canonical_name") or "").casefold()
    )


def bounded(rows: list[dict], *, count: int, audit) -> list[dict]:
    selected: list[dict] = []
    classes: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        klass = str(metadata.get("canonical_class") or "")
        errors = audit(row, forbidden_hashes=set()) if audit is long_direct_audit_errors else audit(row)
        if errors or not answer_matches_target(row) or not klass or classes[klass] >= 2:
            continue
        selected.append(row)
        classes[klass] += 1
        if len(selected) == count:
            return selected
    raise ValueError(f"only selected {len(selected)} of {count} rows")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise RuntimeError(f"immutable trial destination exists: {args.destination}")

    root = args.collection_root
    direct_en = read_jsonl(root / "large-20260818-large-005-direct-option-en-disease" / "accepted.jsonl")
    direct_zh = read_jsonl(root / "large-20260818-large-005-direct-option-zh-disease" / "accepted.jsonl")
    zh_by_image = {str((row.get("metadata") or {}).get("image_sha256") or ""): row for row in direct_zh}
    paired: list[dict] = []
    for row in direct_en:
        image = str((row.get("metadata") or {}).get("image_sha256") or "")
        partner = zh_by_image.get(image)
        if partner is not None:
            paired.extend([row, partner])
    # The current large-005 bilingual intersection has 54 rows after the
    # row-quality and per-class-cap checks.  Keep the smoke deterministic and
    # conservative at 48 paired-route rows rather than weakening its checks.
    direct = bounded(paired, count=48, audit=long_direct_audit_errors)

    rag_source = read_jsonl(root / "large-20260818-large-005-rag-option-en-disease" / "accepted.jsonl")
    rag = bounded(rag_source, count=48, audit=strict_rag_trajectory_errors)
    direct_images = {str((row.get("metadata") or {}).get("image_sha256") or "") for row in direct}
    rag_images = {str((row.get("metadata") or {}).get("image_sha256") or "") for row in rag}
    if direct_images & rag_images:
        raise ValueError("trial Direct/RAG image overlap")

    rows = [*direct, *rag]
    data_hash = canonical_json_hash(rows)
    args.destination.mkdir(parents=True)
    write_jsonl(args.destination / "data.jsonl", rows)
    manifest = {
        "schema_version": "agrinet.reconstructive-sft-freeze/v2",
        "artifact_id": "agrinet-hermes-large005-trial-smoke-v1",
        "stage": "trial_only_not_release",
        "immutable": True,
        "data_sha256": data_hash,
        "source": "large-20260818-large-005 accepted rows",
        "limitations": [
            "not a 560 Direct + 560 RAG freeze",
            "not class/cell complete",
            "not authorized for diagnostic or formal claims",
        ],
        "rows": {"direct": len(direct), "rag": len(rag), "total": len(rows)},
    }
    (args.destination / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (args.destination / "selection.json").write_text(
        json.dumps({"direct_rows": len(direct), "rag_rows": len(rag), "data_sha256": data_hash}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["rows"] | {"data_sha256": data_hash}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
