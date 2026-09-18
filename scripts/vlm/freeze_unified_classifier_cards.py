#!/usr/bin/env python3
"""Freeze Known-only ViT-L Top-5 predictions as unified public cards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from agrinet.vlm.auto_route import classifier_card, contract_hashes


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-prediction-superset", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen cards: {args.output}")
    manifest, predictions = rows(args.manifest), rows(args.predictions)
    labels = json.loads(args.label_map.read_text(encoding="utf-8"))
    names = {str(item["canonical_class_code"]): str(item["english_name"]) for item in labels}
    expected = {str(item["id"]): item for item in manifest}
    by_id = {str(item["image_id"]): item for item in predictions}
    covered = set(expected).issubset(by_id) if args.allow_prediction_superset else set(expected) == set(by_id)
    if len(expected) != len(manifest) or len(by_id) != len(predictions) or not covered:
        raise ValueError("classifier predictions must uniquely cover the public manifest IDs")
    frozen = []
    for item in manifest:
        item_id, prediction = str(item["id"]), by_id[str(item["id"])]
        if prediction.get("image_sha256") != item.get("image_sha256"):
            raise ValueError(f"{item_id}: classifier image SHA mismatch")
        topk = prediction.get("topk") or []
        candidates = [{"name": names[str(candidate["canonical_class_code"])],
                       "score": float(candidate["confidence"])} for candidate in topk]
        frozen.append({"id": item_id, "image_sha256": item["image_sha256"],
                       "predict": classifier_card(candidates),
                       "expand": classifier_card(candidates, expanded=True)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                                    for item in frozen), encoding="utf-8")
    report = {"schema_version": "agrinet.unified-classifier-cards/v1", "rows": len(frozen),
              "manifest_sha256": sha256(args.manifest), "predictions_sha256": sha256(args.predictions),
              "label_map_sha256": sha256(args.label_map), "checkpoint_sha256": sha256(args.checkpoint),
              "cards_sha256": sha256(args.output), **contract_hashes()}
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
