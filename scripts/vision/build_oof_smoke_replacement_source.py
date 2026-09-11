#!/usr/bin/env python3
"""Create a versioned 32-image replacement source without altering v1 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_oof_smoke_source import (CELLS, REGISTRY_SHA, excluded_hashes,
                                   historical_group_exclusions, order, pattern, read_jsonl)


def cell(row: dict) -> tuple[str, str, str]:
    return (str(row["question_type"]), str(row["language"]), str(row["task_domain"]))


def make_sample(row: dict, classes: dict[str, dict], *, question_type: str, language: str) -> dict:
    code = row["canonical_class_code"]
    truth = classes[code]
    domain = row["domain"]
    public_name = truth["canonical_english_name"] if language == "en" else truth["canonical_chinese_name"]
    question = ("Identify the disease or pest shown in the image." if language == "en"
                else "请识别图像中的病害或害虫。")
    public_options: list[dict] = []
    correct = None
    digest = row["image_sha256"]
    if question_type == "option":
        distractors = [item for item in classes.values()
                       if item["class_role"] == "known" and item["domain"] == domain
                       and item["canonical_class_code"] != code]
        options = sorted([truth] + sorted(
            distractors, key=lambda item: order("micu-classifier-hcv-v2-oof-smoke-v2", f"{digest}:{item['canonical_class_code']}"))[:3],
            key=lambda item: order("micu-classifier-hcv-v2-oof-smoke-v2", f"option:{digest}:{item['canonical_class_code']}"))
        public_options = [{"label": chr(65 + index), "code": item["canonical_class_code"],
                           "name": item["canonical_english_name"], "name_zh": item["canonical_chinese_name"]}
                          for index, item in enumerate(options)]
        correct = next(item["label"] for item in public_options if item["code"] == code)
        question += " Choose one option: " + "; ".join(
            f"{item['label']}: {item['name'] if language == 'en' else item['name_zh']}" for item in public_options)
    return {
        "sample_id": "oof-smoke-v2-" + digest[:20], "image_group_id": "image:" + digest,
        "source_group_id": row["source_group_id"], "near_duplicate_group_id": row["near_duplicate_group_id"],
        "image_sha256": digest, "image_path": row["image_path"], "question": question,
        "question_type": question_type, "language": language, "task_domain": domain,
        "dataset_version": "open_agri_v3", "split": "train_candidate", "prediction": row["prediction"],
        "public_options": public_options, "private": {
            "truth_code": code, "class_role": "known", "target_pattern": pattern(row),
            "sampling_bucket": "hard" if pattern(row) in {"P3", "P4", "P5"} else "ordinary",
            "status": "replacement", "intervention": False, "simulated_unknown": False,
            "correct_option": correct, "truth_name": public_name, "mae_saw_related_unlabeled": "unknown",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-source", type=Path, required=True)
    parser.add_argument("--retired", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--class-split", type=Path, required=True)
    parser.add_argument("--images-manifest", type=Path, required=True)
    parser.add_argument("--output-source", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-exclusions", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    args = parser.parse_args()
    source = read_jsonl(args.v1_source)
    retired = read_jsonl(args.retired)
    retired_ids = {str(item["sample_id"]) for item in retired}
    retired_sha = {str(item["image_sha256"]) for item in retired}
    if len(source) != 32 or len(retired_ids) != 3 or not retired_ids <= {str(item["sample_id"]) for item in source}:
        raise ValueError("v1/replacement retirement coverage is invalid")
    retained = [item for item in source if item["sample_id"] not in retired_ids]
    if len(retained) != 29:
        raise ValueError("v2 must retain exactly 29 v1 source rows")
    classes = {str(item["canonical_class_code"]): item for item in read_jsonl(args.class_split)}
    roles = {code: item["class_role"] for code, item in classes.items()}
    blocked = excluded_hashes(args.exclude) | retired_sha | {str(item["image_sha256"]) for item in retained}
    blocked_sources, blocked_groups = historical_group_exclusions(blocked_hashes=blocked, images_manifest=args.images_manifest)
    used_sources = {str(item["source_group_id"]) for item in retained}
    used_groups = {str(item["near_duplicate_group_id"]) for item in retained}
    retired_cells = {str(item["sample_id"]): cell(item) for item in source if item["sample_id"] in retired_ids}
    pool = []
    for row in read_jsonl(args.merged):
        code = str(row.get("canonical_class_code") or "")
        if (roles.get(code) != "known" or row.get("image_sha256") in blocked
                or row.get("source_group_id") in blocked_sources | used_sources
                or row.get("near_duplicate_group_id") in blocked_groups | used_groups):
            continue
        if row.get("prediction", {}).get("kind") != "out_of_fold" or row["prediction"].get("registry_sha256") != REGISTRY_SHA:
            raise ValueError("replacement pool contains invalid OOF provenance")
        pool.append(row)
    replacements = []
    local_used_sha = set(blocked); local_used_sources = set(used_sources); local_used_groups = set(used_groups)
    for retired_id, target in sorted(retired_cells.items()):
        q, lang, domain = target
        candidates = sorted((row for row in pool if row["domain"] == domain
                             and row["image_sha256"] not in local_used_sha
                             and row["source_group_id"] not in local_used_sources
                             and row["near_duplicate_group_id"] not in local_used_groups),
                            key=lambda row: order("micu-classifier-hcv-v2-oof-smoke-v2", f"{retired_id}:{row['image_sha256']}"))
        if not candidates:
            raise ValueError(f"no isolated replacement for {retired_id}")
        row = candidates[0]
        replacement = make_sample(row, classes, question_type=q, language=lang)
        replacements.append(replacement)
        local_used_sha.add(row["image_sha256"]); local_used_sources.add(row["source_group_id"]); local_used_groups.add(row["near_duplicate_group_id"])
    selected = retained + replacements
    counts = Counter(cell(item) for item in selected)
    if len(selected) != 32 or any(counts[item] != 4 for item in CELLS):
        raise ValueError(f"replacement cells invalid: {dict(counts)}")
    if len({item["image_sha256"] for item in selected}) != 32 or len({item["source_group_id"] for item in selected}) != 32 or len({item["near_duplicate_group_id"] for item in selected}) != 32:
        raise ValueError("replacement source lacks image/source/group uniqueness")
    args.output_source.parent.mkdir(parents=True, exist_ok=True)
    args.output_source.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in selected), encoding="utf-8")
    manifest = {
        "schema_version": "agrinet.micu-classifier-hcv-v2-replacement/v1",
        "v1_source": str(args.v1_source), "retired": retired,
        "retained_sample_ids": [item["sample_id"] for item in retained],
        "replacements": [{"retired_sample_id": retired_id, "replacement_sample_id": replacement["sample_id"],
                          "cell": "-".join(retired_cells[retired_id]), "image_sha256": replacement["image_sha256"]}
                         for retired_id, replacement in zip(sorted(retired_cells), replacements)],
        "training_eligible": False,
    }
    args.output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    exclusions = {
        "schema_version": "agrinet.hcv-classifier-replacement-exclusions/v1",
        "complete": True,
        "provenance": {
            "historical_inputs": [str(path) for path in args.exclude],
            "retired_unknown_delivery": str(args.retired),
            "replaced_from": str(args.v1_source),
            "group_expansion": "source_group_id_and_exact_phash_group",
        },
        "retired": retired,
        "referenced_predecessor_sample_ids": [item["sample_id"] for item in retained],
        "replacement_sample_ids": [item["sample_id"] for item in replacements],
    }
    args.output_exclusions.write_text(json.dumps(exclusions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(selected), "retained": len(retained), "replacements": manifest["replacements"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
