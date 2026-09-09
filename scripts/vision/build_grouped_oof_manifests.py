#!/usr/bin/env python3
"""Build deterministic three-fold grouped OOF manifests for OpenAgri v3 Known images."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fold_for(group: str, folds: int) -> int:
    return int(hashlib.sha256(group.encode()).hexdigest()[:16], 16) % folds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    if args.folds != 3:
        raise ValueError("v2 OOF contract requires exactly three folds")
    dataset = args.dataset_root.resolve()
    rows = []
    classes = {}
    with (dataset / "manifests/class_split.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line); classes[str(row["canonical_class_code"])] = row
    with (dataset / "manifests/images.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("image_split") != "train_candidate":
                continue
            if str(row["canonical_class_code"]) not in classes or classes[str(row["canonical_class_code"])].get("class_role") != "known":
                continue
            source = str(Path(str(row["source_path"])).resolve())
            # The frozen manifest has no stronger source-group field. Keep each
            # source path as its own conservative group; exact pHash is the
            # only near-duplicate relation asserted by this builder.
            source_group = "source:" + hashlib.sha256(source.encode()).hexdigest()
            near_group = "phash:" + hashlib.sha256(str(row.get("phash", "")).encode()).hexdigest()
            # A near-duplicate cluster must be indivisible across folds. Source
            # paths are per-file in this frozen manifest and cannot strengthen
            # that constraint, so use the asserted pHash cluster as the fold key.
            group = near_group
            fold = fold_for(group, args.folds)
            rows.append({
                "image_id": row["image_name"], "image_path": str((dataset.parent / "all" / row["image_name"]).resolve()) if not Path(row["source_path"]).is_file() else str(Path(row["source_path"]).resolve()),
                "image_sha256": row["image_sha256"], "phash": row.get("phash", ""),
                "canonical_class_code": row["canonical_class_code"], "domain": row["domain"],
                "label": None, "source_group_id": source_group, "near_duplicate_group_id": near_group,
                "fold": fold,
            })
    known_codes = sorted(code for code, item in classes.items() if item.get("class_role") == "known")
    labels = {code: i for i, code in enumerate(known_codes)}
    for row in rows:
        row["label"] = labels[row["canonical_class_code"]]
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
    train_sha = set()
    with args.training_manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip(): train_sha.add(json.loads(line)["image_sha256"])
    if train_sha != {row["image_sha256"] for row in rows}:
        raise ValueError("OOF source rows do not exactly match the classifier training manifest")
    args.output_root.mkdir(parents=True, exist_ok=False)
    for fold in range(args.folds):
        held = [r for r in rows if r["fold"] == fold]
        train = [r for r in rows if r["fold"] != fold]
        for name, payload in (("train", train), ("heldout", held)):
            path = args.output_root / f"fold-{fold}" / f"{name}.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in sorted(payload, key=lambda x: x["image_sha256"])), encoding="utf-8")
    summary = {
        "schema_version": "agrinet.open-agri-v3.grouped-oof-manifests/v1",
        "folds": args.folds, "rows": len(rows), "known_classes": len(labels),
        "source_group_rule": "source_path_sha256_per_image_conservative",
        "near_duplicate_group_rule": "exact_phash_sha256_only",
        "fold_group_rule": "near_duplicate_group_id",
        "fold_counts": {str(f): len([r for r in rows if r["fold"] == f]) for f in range(args.folds)},
        "class_counts_by_fold": {str(f): dict(Counter(r["canonical_class_code"] for r in rows if r["fold"] == f)) for f in range(args.folds)},
        "training_manifest_sha256": digest(args.training_manifest),
        "dataset_manifest_sha256": digest(dataset / "manifests/images.jsonl"),
        "training_eligible": False,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
