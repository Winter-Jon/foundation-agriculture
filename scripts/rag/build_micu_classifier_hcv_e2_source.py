#!/usr/bin/env python3
"""Freeze the balanced 80-targeted + 80-random E2 exploration source."""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from collections import Counter
from pathlib import Path
from typing import Any

from scipy.optimize import linear_sum_assignment

PATTERNS = tuple(f"P{i}" for i in range(1, 11))
CELLS = (
    "open-en-disease", "open-en-pest", "open-zh-disease", "open-zh-pest",
    "option-en-disease", "option-en-pest", "option-zh-disease", "option-zh-pest",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def cached_sha256(path: Path) -> str:
    return sha256(path)


def by_id(rows: list[dict[str, Any]], *, expected: int, name: str) -> dict[str, dict[str, Any]]:
    result = {str(row["sample_id"]): row for row in rows}
    if len(rows) != expected or len(result) != expected:
        raise ValueError(f"{name} must contain {expected} unique samples")
    return result


def cell(row: dict[str, Any]) -> str:
    return "-".join(str(row[key]) for key in ("question_type", "language", "task_domain"))


def rag_codes(row: dict[str, Any]) -> list[str]:
    if row.get("status") != "success":
        return []
    return [str((item.get("metadata") or {}).get("code") or "")
            for item in (row.get("raw_response") or {}).get("evidence", [])]


def qwen_supports_truth(source: dict[str, Any], qwen: dict[str, Any], class_meta: dict[str, dict[str, Any]]) -> bool:
    truth = class_meta[str(source["private"]["truth_code"])]
    text = str(qwen.get("output_preview") or "").casefold()
    names = {str(truth.get("canonical_english_name") or "").casefold(),
             str(truth.get("canonical_chinese_name") or "").casefold()}
    return any(name and name in text for name in names)


def pattern_score(pattern: str, source: dict[str, Any], qwen: dict[str, Any], rag: dict[str, Any],
                  class_meta: dict[str, dict[str, Any]]) -> float:
    private = source["private"]; candidate = str(private["candidate_pattern"])
    truth = str(private["truth_code"]); top5 = source["prediction"]["top5"]
    top_codes = [str(item["code"]) for item in top5]
    classifier_hit = truth in top_codes; classifier_top = top_codes[0] == truth
    retrieval = rag_codes(rag); rag_hit = truth in retrieval; rag_top = bool(retrieval) and retrieval[0] == truth
    qwen_hit = qwen_supports_truth(source, qwen, class_meta)
    confidence = float(top5[0]["score"])
    stable = int(hashlib.sha256(f"{pattern}:{source['sample_id']}".encode()).hexdigest()[:8], 16) / 2**32
    if pattern in {"P1", "P2", "P3", "P4", "P5", "P6"}:
        score = 100.0 if candidate == pattern else -100.0
    elif pattern == "P7":
        score = 20.0 * ((classifier_top != rag_top) or (qwen_hit != classifier_top)) + 5.0 * rag_hit
    elif pattern == "P8":
        score = 20.0 * (not rag_hit) + 4.0 * (not rag_top) + 2.0 * classifier_hit
    elif pattern == "P9":
        score = 14.0 * (qwen_hit != classifier_top) + 7.0 * (rag_hit and not qwen_hit)
        score += 3.0 * any(word in str(qwen.get("output_preview") or "").casefold()
                           for word in ("however", "difficult", "可能", "无法", "larva", "adult"))
    else:  # P10: all three weak or conflicting, suitable for learning to stop.
        score = 12.0 * (not classifier_top) + 10.0 * (not rag_hit) + 8.0 * (not qwen_hit)
        score += 4.0 * (confidence < 0.5)
    return score + stable * 1e-3


def load_p6_predictions(paths: list[Path]) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = {}
    for group, path in enumerate(paths):
        rows = read_jsonl(path)
        indexed = {str(row["image_sha256"]): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"P6 group {group} predictions contain duplicate images")
        result[group] = indexed
    return result


def p6_prediction(*, source: dict[str, Any], group: int, raw: dict[str, Any],
                  group_root: Path, excluded: list[str], class_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    label_map_path = group_root / "label_map.json"
    checkpoint = group_root / "classifier/model_best.pth.tar"
    train_manifest = group_root / "manifests/train.jsonl"
    labels = json.loads(label_map_path.read_text(encoding="utf-8"))
    by_index = {int(item["index"]): item for item in labels}
    label_codes = [str(item["canonical_class_code"]) for item in labels]
    truth = str(source["private"]["truth_code"])
    top = raw.get("topk")
    if len(labels) != 99 or not isinstance(top, list) or len(top) != 5:
        raise ValueError("P6 prediction requires 99 labels and stored Top-5")
    if truth not in excluded or truth in label_codes:
        raise ValueError("P6 truth leaked into classifier label space")
    rendered = []
    for item in top:
        meta = by_index[int(item["class_index"])]
        code = str(meta["canonical_class_code"])
        if code != str(item["canonical_class_code"]):
            raise ValueError("P6 raw prediction disagrees with label map")
        canonical = class_meta[code]
        rendered.append({"code": code, "name": canonical["canonical_english_name"],
                         "name_zh": canonical["canonical_chinese_name"],
                         "score": float(item["confidence"])})
    if truth in {item["code"] for item in rendered}:
        raise ValueError("P6 truth leaked into Top-5")
    return {
        "kind": "p6_class_holdout", "classifier_version": f"vitl-p6-group{group}-v1",
        "p6_group": group, "image_sha256": source["image_sha256"],
        "excluded_supervised_codes": excluded, "label_codes": label_codes,
        "checkpoint": str(checkpoint), "checkpoint_sha256": cached_sha256(checkpoint),
        "label_map": str(label_map_path), "label_map_sha256": cached_sha256(label_map_path),
        "training_manifest": str(train_manifest), "training_manifest_sha256": cached_sha256(train_manifest),
        "registry_sha256": source["prediction"]["registry_sha256"],
        "mae_saw_related_unlabeled": source["private"].get("mae_saw_related_unlabeled", "unknown"),
        "top5": rendered,
    }


def assign_targeted(rows: list[dict[str, Any]], qwen: dict[str, dict[str, Any]],
                    rag: dict[str, dict[str, Any]], class_meta: dict[str, dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    if len(rows) != 30:
        raise ValueError("each cell must provide 30 targeted pre-screen candidates")
    ordinary = [row for row in rows if row["private"].get("p6_group") is None]
    holdout = [row for row in rows if row["private"].get("p6_group") is not None]
    if len(ordinary) != 25 or len(holdout) != 5:
        raise ValueError("each cell requires 25 ordinary and five P6 candidates")
    selected: list[tuple[dict[str, Any], str]] = []
    for patterns, candidates in ((PATTERNS[:5], ordinary), (PATTERNS[5:], holdout)):
        costs = [[-pattern_score(pattern, row, qwen[row["sample_id"]],
                                 rag[row["sample_id"]], class_meta)
                  for row in candidates] for pattern in patterns]
        pattern_indices, row_indices = linear_sum_assignment(costs)
        selected.extend((candidates[row_index], patterns[pattern_index])
                        for pattern_index, row_index in zip(pattern_indices, row_indices))
    return selected


def final_row(row: dict[str, Any], *, arm: str, pattern: str, qwen: dict[str, Any],
              rag: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["sample_id"] = "e2-" + str(row["image_sha256"])[:20]
    private = dict(row["private"])
    private.update({"sampling_arm": arm, "sampling_bucket": arm, "target_pattern": pattern,
                    "status": "selected_for_exploration",
                    "prescreen_signals": {
                        "qwen_sample_id": qwen["sample_id"],
                        "rag_sample_id": rag["sample_id"],
                        "rag_codes": rag_codes(rag),
                    }})
    result["private"] = private
    result["training_eligible"] = False
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, action="append", required=True)
    parser.add_argument("--rag", type=Path, required=True)
    parser.add_argument("--class-split", type=Path, required=True)
    parser.add_argument("--p6-groups", type=Path, required=True)
    parser.add_argument("--p6-prediction", type=Path, action="append", required=True)
    parser.add_argument("--output-source", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-exclusions", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = read_jsonl(args.source)
    source_by_id = by_id(source, expected=320, name="source")
    qwen_rows = [row for path in args.qwen for row in read_jsonl(path)]
    qwen = by_id(qwen_rows, expected=320, name="Qwen pre-screen")
    rag = by_id(read_jsonl(args.rag), expected=320, name="RAG pre-screen")
    if set(source_by_id) != set(qwen) or set(source_by_id) != set(rag):
        raise ValueError("pre-screen sample identities do not match")
    class_meta = {str(row["canonical_class_code"]): row for row in read_jsonl(args.class_split)}
    groups = json.loads(args.p6_groups.read_text(encoding="utf-8"))["groups"]
    if len(groups) != 3 or len(args.p6_prediction) != 3:
        raise ValueError("three P6 groups and prediction files are required")
    p6_raw = load_p6_predictions(args.p6_prediction)

    for row in source:
        group = row["private"].get("p6_group")
        if group is None:
            continue
        group = int(group)
        raw = p6_raw[group].get(str(row["image_sha256"]))
        if raw is None:
            raise ValueError(f"missing P6 prediction for {row['sample_id']}")
        group_root = Path(groups[group]["artifact_root"])
        excluded = [str(code) for code in groups[group]["excluded_supervised_codes"]]
        row["prediction"] = p6_prediction(source=row, group=group, raw=raw,
                                               group_root=group_root, excluded=excluded,
                                               class_meta=class_meta)
        row["private"]["simulated_unknown"] = True

    selected: list[dict[str, Any]] = []
    for fixed_cell in CELLS:
        rows = [row for row in source if cell(row) == fixed_cell]
        targeted = [row for row in rows if row["private"]["sampling_arm"] == "targeted"]
        random = [row for row in rows if row["private"]["sampling_arm"] == "random"]
        if len(random) != 10:
            raise ValueError(f"{fixed_cell} must provide exactly ten random controls")
        for row, pattern in assign_targeted(targeted, qwen, rag, class_meta):
            selected.append(final_row(row, arm="targeted", pattern=pattern,
                                      qwen=qwen[row["sample_id"]], rag=rag[row["sample_id"]]))
        for row in random:
            pattern = str(row["private"]["candidate_pattern"])
            selected.append(final_row(row, arm="random", pattern=pattern,
                                      qwen=qwen[row["sample_id"]], rag=rag[row["sample_id"]]))

    identity_keys = ("sample_id", "image_sha256", "image_group_id",
                     "source_group_id", "near_duplicate_group_id")
    if len(selected) != 160:
        raise ValueError("final E2 source must contain 160 rows")
    for key in identity_keys:
        if len({str(row[key]) for row in selected}) != 160:
            raise ValueError(f"final E2 source is not independent by {key}")
    cells = Counter(cell(row) for row in selected)
    arms = Counter(row["private"]["sampling_arm"] for row in selected)
    targeted_patterns = Counter(row["private"]["target_pattern"] for row in selected
                                if row["private"]["sampling_arm"] == "targeted")
    if cells != {name: 20 for name in CELLS}:
        raise ValueError("final E2 source must contain twenty rows per cell")
    if arms != {"targeted": 80, "random": 80}:
        raise ValueError("final E2 arms must be 80 targeted + 80 random")
    if targeted_patterns != {pattern: 8 for pattern in PATTERNS}:
        raise ValueError("targeted arm must assign eight independent rows per pattern")

    selected.sort(key=lambda row: (cell(row), row["private"]["sampling_arm"],
                                   row["private"]["target_pattern"], row["sample_id"]))
    args.output_source.parent.mkdir(parents=True, exist_ok=True)
    args.output_source.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "agrinet.micu-classifier-hcv-e2-source/v1",
        "source": str(args.output_source), "source_sha256": sha256(args.output_source),
        "rows": len(selected), "cells": dict(sorted(cells.items())),
        "arms": dict(sorted(arms.items())),
        "targeted_primary_patterns": dict(sorted(targeted_patterns.items())),
        "prediction_kinds": dict(sorted(Counter(row["prediction"]["kind"] for row in selected).items())),
        "p6_semantics": "classifier excludes held-out classes; complete local AgriNet RAG remains visible",
        "teacher_visibility": "sampling signals and private patterns are never included in public_sample",
        "training_eligible": False,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_exclusions is not None:
        chosen = {row["image_sha256"] for row in selected}
        excluded = {
            "schema_version": "agrinet.micu-classifier-hcv-e2-exclusions/v1",
            "candidate_rows": len(source), "selected_rows": len(selected),
            "excluded_rows": len(source) - len(selected),
            "reason": "not_selected_by_frozen_80_targeted_plus_80_random_assignment",
            "image_sha256": sorted(row["image_sha256"] for row in source if row["image_sha256"] not in chosen),
            "training_eligible": False,
        }
        args.output_exclusions.parent.mkdir(parents=True, exist_ok=True)
        args.output_exclusions.write_text(
            json.dumps(excluded, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
