#!/usr/bin/env python3
"""Materialize one grouped OOF fold for the existing classifier workflow."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-manifests", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.fold_manifests / f"fold-{args.fold}"
    train = [json.loads(x) for x in (source / "train.jsonl").read_text().splitlines() if x]
    held = [json.loads(x) for x in (source / "heldout.jsonl").read_text().splitlines() if x]
    labels = json.loads(args.label_map.read_text(encoding="utf-8"))
    root = args.output_root / f"fold-{args.fold}"
    (root / "manifests").mkdir(parents=True, exist_ok=False)
    (root / "manifests/train.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in train), encoding="utf-8")
    (root / "manifests/dev_known.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in held), encoding="utf-8")
    (root / "label_map.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "agrinet.open-agri-v3.oof-fold-artifact/v1",
        "fold": args.fold, "train_rows": len(train), "heldout_rows": len(held),
        "train_sha256": sha256(root / "manifests/train.jsonl"),
        "heldout_sha256": sha256(root / "manifests/dev_known.jsonl"),
        "label_map_sha256": sha256(root / "label_map.json"),
        "training_eligible": False,
        "prediction_kind": "out_of_fold",
        "mae_saw_related_unlabeled": "unknown",
    }
    (root / "fold_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
