#!/usr/bin/env python3
"""Audit a merged grouped-OOF prediction manifest against its source folds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--expected-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.expected_summary.read_text(encoding="utf-8"))
    merged = read(args.merged)
    by_sha = {row["image_sha256"]: row for row in merged}
    errors: list[str] = []
    if len(merged) != expected["rows"] or len(by_sha) != expected["rows"]:
        errors.append("merged predictions do not contain each expected image exactly once")
    fold_groups: dict[str, set[int]] = {}
    source_rows = 0
    for fold in range(3):
        held = read(args.manifest_root / f"fold-{fold}/manifests/dev_known.jsonl")
        checkpoint = args.manifest_root / f"fold-{fold}" / "classifier" / "model_best.pth.tar"
        label_map = args.manifest_root / f"fold-{fold}" / "label_map.json"
        fold_checkpoint_sha256 = sha256(checkpoint) if checkpoint.is_file() else None
        fold_label_map_sha256 = sha256(label_map) if label_map.is_file() else None
        fold_training_cache: dict[str, tuple[bool, dict[str, set[str]]]] = {}
        source_rows += len(held)
        for row in held:
            fold_groups.setdefault(row["near_duplicate_group_id"], set()).add(fold)
            prediction = by_sha.get(row["image_sha256"])
            if prediction is None:
                errors.append(f"missing prediction: {row['image_sha256']}")
                continue
            provenance = prediction.get("prediction", {})
            if (provenance.get("held_out_fold") != fold or
                    provenance.get("folds") != 3 or
                    provenance.get("kind") != "out_of_fold"):
                errors.append(f"wrong OOF provenance: {row['image_sha256']}")
            train_path = Path(provenance.get("training_manifest", ""))
            train_key = str(train_path)
            if train_key not in fold_training_cache:
                if not train_path.is_file() or sha256(train_path) != provenance.get("training_manifest_sha256"):
                    fold_training_cache[train_key] = (False, {})
                else:
                    training = read(train_path)
                    fold_training_cache[train_key] = (
                        True,
                        {key: {item[key] for item in training}
                         for key in ("image_sha256", "source_group_id", "near_duplicate_group_id")},
                    )
            training_valid, identities = fold_training_cache[train_key]
            if not training_valid:
                errors.append(f"training manifest provenance mismatch: {row['image_sha256']}")
            elif any(row[key] in identities[key] for key in identities):
                errors.append(f"training overlap: {row['image_sha256']}")
            if fold_checkpoint_sha256 is None or provenance.get("checkpoint_sha256") != fold_checkpoint_sha256:
                errors.append(f"checkpoint provenance mismatch: {row['image_sha256']}")
            if fold_label_map_sha256 is None or provenance.get("label_map_sha256") != fold_label_map_sha256:
                errors.append(f"label-map provenance mismatch: {row['image_sha256']}")
            top5 = provenance.get("top5")
            if (not isinstance(top5, list) or len(top5) != 5 or
                    len({item.get("code") for item in top5 if isinstance(item, dict)}) != 5 or
                    any(not isinstance(item, dict) or not isinstance(item.get("score"), (int, float))
                        for item in top5) or
                    any(top5[i]["score"] < top5[i + 1]["score"] for i in range(4))):
                errors.append(f"invalid Top-5: {row['image_sha256']}")
    if source_rows != expected["rows"]:
        errors.append("fold held-out rows do not cover expected source rows")
    if any(len(folds) != 1 for folds in fold_groups.values()):
        errors.append("near-duplicate group crosses folds")
    report = {"schema_version": "agrinet.open-agri-v3.grouped-oof-audit/v1", "source_rows": source_rows, "merged_rows": len(merged), "unique_near_duplicate_groups": len(fold_groups), "errors": errors, "ready": not errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
