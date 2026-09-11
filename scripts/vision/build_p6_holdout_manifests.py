#!/usr/bin/env python3
"""Build three deterministic classifier-only class holdout groups for E2 P6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from agrinet.vision.workflow import build_manifests


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def select_groups(classes: list[dict], *, seed: str) -> list[list[str]]:
    groups = [[] for _ in range(3)]
    for domain in ("disease", "pest"):
        eligible = [row for row in classes if row.get("class_role") == "known"
                    and row.get("domain") == domain and int(row.get("train_candidate_images") or 0) >= 100]
        eligible.sort(key=lambda row: stable(seed, str(row["canonical_class_code"])))
        if len(eligible) < 12:
            raise ValueError(f"P6 needs at least 12 eligible {domain} classes")
        for index, row in enumerate(eligible[:12]):
            groups[index % 3].append(str(row["canonical_class_code"]))
    if any(len(group) != 8 for group in groups) or len(set().union(*map(set, groups))) != 24:
        raise ValueError("P6 holdout groups must be three disjoint eight-class sets")
    return [sorted(group) for group in groups]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", default="micu-classifier-hcv-e2-p6-v1")
    args = parser.parse_args()
    classes = read_jsonl(args.dataset_root / "manifests/class_split.jsonl")
    images = read_jsonl(args.dataset_root / "manifests/images.jsonl")
    groups = select_groups(classes, seed=args.seed)
    summary = {"schema_version": "agrinet.micu-classifier-hcv-e2-p6-groups/v1",
               "seed": args.seed, "groups": [], "training_eligible": False}
    for index, codes in enumerate(groups):
        root = args.output_root / f"group-{index}"
        build_manifests(args.dataset_root, root, set(codes), include_mae=False)
        label_index = {row["canonical_class_code"]: row["index"]
                       for row in json.loads((root / "label_map.json").read_text(encoding="utf-8"))}
        holdout = []
        for row in images:
            if row.get("image_split") != "train_candidate" or row.get("canonical_class_code") not in codes:
                continue
            holdout.append({"image_id": row["image_name"], "image_path": str(Path(row["image_path"]).resolve()),
                            "image_sha256": row["image_sha256"], "label": 0,
                            "canonical_class_code": row["canonical_class_code"], "domain": row["domain"],
                            "source_group_id": row.get("source_group_id"),
                            "near_duplicate_group_id": row.get("near_duplicate_group_id")})
        holdout.sort(key=lambda row: (row["canonical_class_code"], row["image_sha256"]))
        target = root / "manifests/holdout_train_candidate.jsonl"
        target.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in holdout), encoding="utf-8")
        if any(code in label_index for code in codes):
            raise ValueError("held-out class leaked into classifier label map")
        summary["groups"].append({"group": index, "excluded_supervised_codes": codes,
                                  "classifier_classes": len(label_index), "holdout_images": len(holdout),
                                  "artifact_root": str(root), "holdout_manifest": str(target)})
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "groups.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
