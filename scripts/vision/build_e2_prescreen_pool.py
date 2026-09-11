#!/usr/bin/env python3
"""Build the isolated 320-image E2 local pre-screen pool."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_oof_smoke_source import CELLS, excluded_hashes, historical_group_exclusions, read_jsonl


def read_source_rows(path: Path) -> list[dict]:
    """Read an E2 source in either JSONL or compact immutable JSON form."""
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rows") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"exclude source lacks rows: {path}")
    return [row for row in rows if isinstance(row, dict)]


def load_classes(*, class_split: Path, known_label_map: Path | None) -> dict[str, dict]:
    """Load the historical split or normalize registry + frozen known labels."""
    if known_label_map is None:
        return {str(row["canonical_class_code"]): row for row in read_jsonl(class_split)}
    known = {str(row.get("canonical_class_code") or "")
             for row in json.loads(known_label_map.read_text(encoding="utf-8"))
             if isinstance(row, dict)}
    registry = {str(row.get("canonical_code") or ""): row for row in read_jsonl(class_split)}
    if len(known) != 107 or not known.issubset(registry):
        raise ValueError("known-label-map is not a complete canonical registry subset")
    return {code: {"canonical_class_code": code, "code": code,
                   "class_role": "known", "domain": row["domain"],
                   "canonical_english_name": row["canonical_english_name"],
                   "canonical_chinese_name": row["canonical_chinese_name"]}
            for code, row in registry.items() if code in known}


def stable(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def standard_bucket(row: dict) -> str:
    truth = row["canonical_class_code"]
    top = row["prediction"]["top5"]
    rank = next((index for index, item in enumerate(top, 1) if item["code"] == truth), 0)
    if rank == 1 and float(top[0]["score"]) >= 0.8:
        return "P1"
    if rank == 1:
        return "P2"
    if rank in (2, 3):
        return "P3"
    if rank in (4, 5):
        return "P4"
    return "P5"


def make_sample(row: dict, classes: dict[str, dict], *, question_type: str, language: str,
                arm: str, candidate_pattern: str, p6_group: int | None, seed: str) -> dict:
    code = str(row["canonical_class_code"])
    truth = classes[code]
    digest = str(row["image_sha256"])
    domain = str(row["domain"])
    question = "Identify the disease or pest shown in the image." if language == "en" else "请识别图像中的病害或害虫。"
    public_options: list[dict] = []
    correct_option = None
    if question_type == "option":
        by_code = {item["canonical_class_code"]: item for item in classes.values()
                   if item["class_role"] == "known" and item["domain"] == domain}
        distractor_codes = [item["code"] for item in row["prediction"]["top5"]
                            if item["code"] != code and item["code"] in by_code]
        fallback = sorted((candidate for candidate in by_code if candidate != code and candidate not in distractor_codes),
                          key=lambda candidate: stable(seed, f"distractor:{digest}:{candidate}"))
        option_codes = [code] + (distractor_codes + fallback)[:3]
        option_codes.sort(key=lambda candidate: stable(seed, f"option:{digest}:{candidate}"))
        public_options = [{"label": chr(65 + index), "code": candidate,
                           "name": by_code[candidate]["canonical_english_name"],
                           "name_zh": by_code[candidate]["canonical_chinese_name"]}
                          for index, candidate in enumerate(option_codes)]
        correct_option = next(item["label"] for item in public_options if item["code"] == code)
        question += " Choose one option: " if language == "en" else " 请选择一个选项："
        question += "; ".join(f"{item['label']}: {item['name'] if language == 'en' else item['name_zh']}"
                              for item in public_options)
    truth_name = truth["canonical_english_name"] if language == "en" else truth["canonical_chinese_name"]
    return {
        "sample_id": "e2-prescreen-" + digest[:20], "image_group_id": "image:" + digest,
        "source_group_id": row["source_group_id"], "near_duplicate_group_id": row["near_duplicate_group_id"],
        "image_sha256": digest, "image_path": row["image_path"], "question": question,
        "question_type": question_type, "language": language, "task_domain": domain,
        "dataset_version": "open_agri_v3", "split": "train_candidate", "prediction": row["prediction"],
        "public_options": public_options,
        "private": {"truth_code": code, "truth_name": truth_name, "class_role": "known",
                    "sampling_arm": arm, "candidate_pattern": candidate_pattern,
                    "target_pattern": candidate_pattern, "p6_group": p6_group,
                    "sampling_bucket": "random" if arm == "random" else "targeted",
                    "status": "prescreen", "intervention": False,
                    "simulated_unknown": p6_group is not None,
                    "correct_option": correct_option, "mae_saw_related_unlabeled": "unknown"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--class-split", type=Path, required=True,
                        help="Historical class split, or canonical registry with --known-label-map.")
    parser.add_argument("--known-label-map", type=Path,
                        help="Frozen 107-class OOF label map when class-split is canonical registry.")
    parser.add_argument("--images-manifest", type=Path, required=True)
    parser.add_argument("--p6-groups", type=Path, required=True)
    parser.add_argument("--output-source", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--seed", default="micu-classifier-hcv-e2-prescreen-v1")
    parser.add_argument("--target-patterns", default="P1,P2,P3,P4,P5,P6",
                        help="Comma-separated targeted candidate patterns; default preserves the v1 pool.")
    parser.add_argument("--target-per-pattern-per-cell", type=int, default=5,
                        help="Fresh identities per requested pattern in each fixed cell.")
    parser.add_argument("--random-per-cell", type=int, default=10,
                        help="Additional deterministic random identities per fixed cell.")
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--exclude-source", type=Path, action="append", default=[],
                        help="Immutable JSON/JSONL source whose image/source/near-duplicate IDs are blocked.")
    args = parser.parse_args()
    target_patterns = tuple(item.strip() for item in args.target_patterns.split(",") if item.strip())
    allowed_patterns = {"P1", "P2", "P3", "P4", "P5", "P6"}
    if not target_patterns or any(item not in allowed_patterns for item in target_patterns):
        raise ValueError("target-patterns must be a non-empty subset of P1,P2,P3,P4,P5,P6")
    if len(set(target_patterns)) != len(target_patterns):
        raise ValueError("target-patterns must not contain duplicates")
    if args.target_per_pattern_per_cell < 1 or args.random_per_cell < 0:
        raise ValueError("target and random per-cell quotas must be non-negative, with target positive")
    classes = load_classes(class_split=args.class_split, known_label_map=args.known_label_map)
    groups_payload = json.loads(args.p6_groups.read_text(encoding="utf-8"))
    p6_groups = [set(item["excluded_supervised_codes"]) for item in groups_payload["groups"]]
    if len(p6_groups) != 3 or len(set().union(*p6_groups)) != 24:
        raise ValueError("P6 groups must contain 24 mutually exclusive holdout classes")
    holdout_to_group = {code: index for index, group in enumerate(p6_groups) for code in group}
    blocked = excluded_hashes(args.exclude)
    blocked_sources_direct, blocked_groups_direct = set(), set()
    for path in args.exclude_source:
        for row in read_source_rows(path):
            image, source, near = (str(row.get(key) or "") for key in ("image_sha256", "source_group_id", "near_duplicate_group_id"))
            if image:
                blocked.add(image)
            if source:
                blocked_sources_direct.add(source)
            if near:
                blocked_groups_direct.add(near)
    for path in (Path("outputs/artifacts/micu-classifier-hcv-v2/smoke-v1/source.jsonl"),
                 Path("outputs/artifacts/micu-classifier-hcv-v2/smoke-v2/source.jsonl")):
        blocked |= excluded_hashes([path])
    blocked_sources, blocked_groups = historical_group_exclusions(
        blocked_hashes=blocked, images_manifest=args.images_manifest)
    blocked_sources |= blocked_sources_direct
    blocked_groups |= blocked_groups_direct
    pool = []
    for row in read_jsonl(args.merged):
        if row.get("image_sha256") in blocked or row.get("source_group_id") in blocked_sources or row.get("near_duplicate_group_id") in blocked_groups:
            continue
        if row.get("prediction", {}).get("kind") != "out_of_fold":
            continue
        if str(row.get("canonical_class_code")) not in classes or classes[str(row["canonical_class_code"])].get("class_role") != "known":
            continue
        item = dict(row)
        item["standard_bucket"] = standard_bucket(row)
        item["p6_group"] = holdout_to_group.get(str(row["canonical_class_code"]))
        pool.append(item)
    used_sha: set[str] = set(blocked)
    used_sources: set[str] = set()
    used_groups: set[str] = set()
    selected: list[dict] = []
    targeted_counts: Counter[str] = Counter()
    random_counts: Counter[str] = Counter()
    for question_type, language, domain in CELLS:
        candidates = [row for row in pool if row["domain"] == domain and row["image_sha256"] not in used_sha
                      and row["source_group_id"] not in used_sources and row["near_duplicate_group_id"] not in used_groups]
        candidates.sort(key=lambda row: stable(args.seed, f"{question_type}:{language}:{row['image_sha256']}"))
        for bucket in target_patterns:
            eligible = [row for row in candidates if row["standard_bucket"] == bucket]
            if bucket == "P6":
                eligible = [row for row in candidates if row["p6_group"] is not None]
            if len(eligible) < args.target_per_pattern_per_cell:
                raise ValueError(f"{domain} {bucket} lacks {args.target_per_pattern_per_cell} candidates for {question_type}/{language}")
            for row in eligible[:args.target_per_pattern_per_cell]:
                sample = make_sample(row, classes, question_type=question_type, language=language,
                                     arm="targeted", candidate_pattern=bucket, p6_group=None, seed=args.seed)
                if bucket == "P6":
                    sample["private"]["p6_group"] = row["p6_group"]
                selected.append(sample); used_sha.add(row["image_sha256"]); used_sources.add(row["source_group_id"]); used_groups.add(row["near_duplicate_group_id"]); targeted_counts[bucket] += 1
                candidates.remove(row)
        random_candidates = [row for row in candidates if row["image_sha256"] not in used_sha and row["source_group_id"] not in used_sources and row["near_duplicate_group_id"] not in used_groups]
        random_candidates.sort(key=lambda row: stable(args.seed, f"random:{question_type}:{language}:{row['image_sha256']}"))
        if len(random_candidates) < args.random_per_cell:
            raise ValueError(f"{domain} random pool lacks {args.random_per_cell} candidates for {question_type}/{language}")
        for row in random_candidates[:args.random_per_cell]:
            bucket = row["standard_bucket"]
            sample = make_sample(row, classes, question_type=question_type, language=language,
                                 arm="random", candidate_pattern=bucket, p6_group=None, seed=args.seed)
            selected.append(sample); used_sha.add(row["image_sha256"]); used_sources.add(row["source_group_id"]); used_groups.add(row["near_duplicate_group_id"]); random_counts["random"] += 1
    expected_rows = len(CELLS) * (len(target_patterns) * args.target_per_pattern_per_cell + args.random_per_cell)
    if len(selected) != expected_rows or len({row["image_sha256"] for row in selected}) != expected_rows or len({row["source_group_id"] for row in selected}) != expected_rows or len({row["near_duplicate_group_id"] for row in selected}) != expected_rows:
        raise ValueError(f"E2 pool does not have {expected_rows} independent identities")
    args.output_source.parent.mkdir(parents=True, exist_ok=True)
    args.output_source.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
    summary = {"schema_version": "agrinet.micu-classifier-hcv-e2-prescreen/v1", "rows": len(selected),
               "per_cell": dict(Counter(f"{r['question_type']}-{r['language']}-{r['task_domain']}" for r in selected)),
               "arms": dict(Counter(r["private"]["sampling_arm"] for r in selected)),
               "targeted_candidate_patterns": dict(targeted_counts), "random_rows": dict(random_counts),
               "target_patterns": list(target_patterns),
               "target_per_pattern_per_cell": args.target_per_pattern_per_cell,
               "random_per_cell": args.random_per_cell,
               "p6_semantics": "classifier_label_space_excludes_holdout; full_local_rag_registry_remains_visible",
               "training_eligible": False, "exclusions": [str(path) for path in args.exclude],
               "exclude_sources": [str(path) for path in args.exclude_source],
               "known_label_map": str(args.known_label_map) if args.known_label_map else None}
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
