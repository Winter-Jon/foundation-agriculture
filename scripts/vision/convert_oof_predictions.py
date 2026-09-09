#!/usr/bin/env python3
"""Attach auditable provenance to held-out classifier predictions and merge folds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REGISTRY_SHA = "4fa6426203c64631315a2d754f3d53d6c3a7639e26c1c8617b2cbf92caf00cb8"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def convert_fold(root: Path, fold: int) -> tuple[list[dict], dict]:
    fold_root = root / f"fold-{fold}"
    raw_path = fold_root / "classifier" / "predictions_dev_known.jsonl"
    held_path = fold_root / "manifests" / "dev_known.jsonl"
    training_path = fold_root / "manifests" / "train.jsonl"
    checkpoint = fold_root / "classifier" / "model_best.pth.tar"
    label_map = fold_root / "label_map.json"
    required = (raw_path, held_path, training_path, checkpoint, label_map)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("fold prerequisites missing: " + ", ".join(missing))
    raw, held = rows(raw_path), rows(held_path)
    by_sha = {item["image_sha256"]: item for item in held}
    if len(raw) != len(held) or {item["image_sha256"] for item in raw} != set(by_sha):
        raise ValueError(f"fold {fold}: prediction/held-out identity mismatch")
    labels = json.loads(label_map.read_text(encoding="utf-8"))
    names = {str(item["canonical_class_code"]): {
        "name": str(item["english_name"]), "name_zh": str(item["chinese_name"]),
    } for item in labels}
    checkpoint_sha, train_sha, label_sha = sha256(checkpoint), sha256(training_path), sha256(label_map)
    converted = []
    for item in raw:
        meta = by_sha[item["image_sha256"]]
        top5 = [{"code": candidate["canonical_class_code"], "score": float(candidate["confidence"]),
                 **names[str(candidate["canonical_class_code"])]}
                for candidate in item["topk"]]
        if len(top5) != 5 or len({candidate["code"] for candidate in top5}) != 5:
            raise ValueError(f"fold {fold}: invalid Top-5 for {item['image_sha256']}")
        converted.append({
            **meta,
            "prediction": {
                "kind": "out_of_fold", "image_sha256": item["image_sha256"],
                "top5": top5, "classifier_version": "vitl-oof-v2",
                "checkpoint_sha256": checkpoint_sha, "label_map_sha256": label_sha,
                "training_manifest": str(training_path.resolve()),
                "training_manifest_sha256": train_sha, "registry_sha256": REGISTRY_SHA,
                "folds": 3, "held_out_fold": fold,
                "excluded_supervised_codes": [],
                "mae_saw_related_unlabeled": "unknown",
            },
        })
    converted.sort(key=lambda item: item["image_sha256"])
    out = fold_root / "classifier" / "predictions_oof_v2.jsonl"
    out.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in converted), encoding="utf-8")
    return converted, {"fold": fold, "rows": len(converted), "checkpoint_sha256": checkpoint_sha, "training_manifest_sha256": train_sha, "output": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--merge-output", type=Path)
    args = parser.parse_args()
    folds = (args.fold,) if args.fold is not None else (0, 1, 2)
    merged = []
    summaries = []
    for fold in folds:
        converted, summary = convert_fold(args.root, fold)
        merged.extend(converted); summaries.append(summary)
    if args.merge_output:
        identities = [item["image_sha256"] for item in merged]
        if len(identities) != len(set(identities)):
            raise ValueError("OOF merge contains duplicate image identities")
        args.merge_output.parent.mkdir(parents=True, exist_ok=True)
        args.merge_output.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in sorted(merged, key=lambda item: item["image_sha256"])), encoding="utf-8")
    print(json.dumps({"folds": summaries, "merged_rows": len(merged)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
