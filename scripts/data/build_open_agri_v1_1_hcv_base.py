#!/usr/bin/env python3
"""Build the immutable HCV-oriented open_agri_v1.1 intermediate view.

The builder leaves open_agri_v1 untouched. It creates a self-describing image
layout plus VLM candidate/evaluation manifests, with content hashes,
class-role decisions, and a bounded perceptual-duplicate audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v1"
DEFAULT_OUTPUT = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v1_1_hcv_base"
DEFAULT_SIMILAR = REPO_ROOT / "outputs/artifacts/datasets/m1-direct-current-hcv-v1/similar_classes.jsonl"
DEFAULT_RAW_BASE = REPO_ROOT / (
    "outputs/runs/vlm/vlm-hcv-v12-direct-anchor-mix-base-checkpoint232-full-eval-v2/"
    "20260825T230259-2ef993b1-a01/artifacts/epoch-0-checkpoint-232/"
    "formal-direct/artifacts/raw_base_direct/scored.jsonl"
)
DEFAULT_PHASH_CACHE = REPO_ROOT / "outputs/artifacts/datasets/open-agri-v2-phash-cache-v1.jsonl"
SEED = "open-agri-v1.1-hcv-base-20260902"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
PHASH_SIZE = 16
PHASH_THRESHOLD = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--similar-classes", type=Path, default=DEFAULT_SIMILAR)
    parser.add_argument("--raw-base-scored", type=Path, default=DEFAULT_RAW_BASE)
    parser.add_argument("--allow-partial-raw-base", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--phash-threshold", type=int, default=PHASH_THRESHOLD)
    parser.add_argument("--phash-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--phash-cache", type=Path, default=DEFAULT_PHASH_CACHE)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def relative_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(os.path.relpath(source, target.parent))


def dct_matrix(size: int) -> np.ndarray:
    index = np.arange(size)
    matrix = np.cos(np.pi * (2 * index[:, None] + 1) * index[None, :] / (2 * size))
    matrix[0] /= np.sqrt(2)
    return matrix * np.sqrt(2 / size)


DCT_32 = dct_matrix(32)


def phash(path: Path) -> str:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    transformed = DCT_32 @ pixels @ DCT_32.T
    block = transformed[:PHASH_SIZE, :PHASH_SIZE]
    median = np.median(block[1:, :])
    return "".join("1" if value > median else "0" for value in block.ravel())


def hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def deterministic_key(*values: str) -> str:
    return hashlib.sha256((SEED + ":" + ":".join(values)).encode()).hexdigest()


def raw_base_recalls(path: Path, codes: set[str], allow_partial: bool) -> tuple[dict[str, float | None], dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"raw-base scored file does not exist: {path}")
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for row in read_jsonl(path):
        code = str(row.get("label_code") or "")
        if code not in codes:
            continue
        counts[code] += 1
        correct[code] += bool(row.get("correct"))
    missing = sorted(codes - set(counts))
    if missing and not allow_partial:
        raise ValueError(
            f"raw-base coverage is incomplete ({len(counts)}/{len(codes)} classes); "
            "provide a complete per-class scored file or pass --allow-partial-raw-base"
        )
    recalls = {code: (correct[code] / counts[code] if counts[code] else None) for code in codes}
    report = {
        "path": str(path), "sha256": digest(path), "classes_covered": len(counts),
        "classes_missing": missing, "rows_used": sum(counts.values()),
        "partial_coverage_allowed": allow_partial,
    }
    return recalls, report


def load_similar(path: Path, codes: set[str]) -> dict[str, list[str]]:
    rows = {str(row.get("class_code") or ""): row for row in read_jsonl(path)}
    missing = sorted(codes - set(rows))
    if missing:
        raise ValueError(f"similar-class manifest lacks classes: {missing[:10]}")
    values: dict[str, list[str]] = {}
    for code in codes:
        neighbours = [str(value) for value in rows[code].get("hard_negative_codes") or []]
        if len(neighbours) < 3 or any(value not in codes for value in neighbours):
            raise ValueError(f"invalid hard-negative list for {code}")
        values[code] = neighbours
    return values


def read_coverage(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8") as handle:
        return {row["class_code"]: int(row["train"]) for row in csv.DictReader(handle)}


def choose_unknowns(
    codes: set[str], domains: dict[str, str], train_counts: dict[str, int], recalls: dict[str, float | None],
    similar: dict[str, list[str]], targets: dict[str, int],
) -> tuple[set[str], dict[str, str]]:
    """Select hard long-tail unknowns while preserving a known Top-3 bridge."""
    # All known classes must contribute five dev images, so insufficient
    # training capacity is a non-negotiable unknown-class condition.
    unknown = {code for code in codes if train_counts[code] < 5}
    reasons = {
        code: "no_train_candidate" if train_counts[code] == 0 else "insufficient_train_candidates_for_known_dev"
        for code in unknown
    }
    for domain, target in targets.items():
        domain_codes = sorted(code for code in codes if domains[code] == domain)
        fixed = [code for code in unknown if domains[code] == domain]
        needed = target - len(fixed)
        if needed < 0:
            raise ValueError(f"fixed unknowns exceed target for {domain}")
        ranked = sorted(
            (code for code in domain_codes if code not in unknown),
            key=lambda code: (
                0 if recalls[code] is not None else 1,
                -(1 - recalls[code]) if recalls[code] is not None else 0,
                train_counts[code], deterministic_key("unknown", code),
            ),
        )
        selected: list[str] = []
        for code in ranked:
            if len(selected) == needed:
                break
            provisional_unknown = unknown | set(selected) | {code}
            known = codes - provisional_unknown
            if all(any(neighbour in known for neighbour in similar[item]) for item in provisional_unknown):
                selected.append(code)
        if len(selected) != needed:
            raise ValueError(f"could not select {needed} bridgeable unknown classes for {domain}")
        unknown.update(selected)
        for code in selected:
            reasons[code] = "raw_base_hard_then_long_tail" if recalls[code] is not None else "raw_base_missing_long_tail"
    known = codes - unknown
    broken = [code for code in unknown if not any(value in known for value in similar[code])]
    if broken:
        raise ValueError(f"unknown classes without known Top-3 bridge: {broken}")
    return unknown, reasons


def image_rows(manifest: Path, split: str) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(manifest):
        source = Path(str(row["source_path"]))
        if not source.is_file():
            raise FileNotFoundError(source)
        rows.append({
            "class_code": str(row["class_code"]), "domain": str(row["domain"]),
            "image_name": str(row["image_name"]), "source_path": str(source),
            "sha256": str(row.get("sha256") or digest(source)), "image_split": split,
        })
    return rows


def deduplicate_content(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retain the first deterministic occurrence of each content hash per split."""
    retained: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        first = seen.get(row["sha256"])
        if first is None:
            seen[row["sha256"]] = row
            retained.append(row)
            continue
        audit.append({
            "sha256": row["sha256"], "split": split, "class_code": row["class_code"],
            "image_name": row["image_name"], "source_path": row["source_path"],
            "retained_source_path": first["source_path"], "resolution": "deduplicated_within_input_split",
        })
    return retained, audit


def choose_dev_images(train_rows: list[dict[str, Any]], dev_codes: set[str]) -> tuple[list[dict[str, Any]], set[str]]:
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        if row["class_code"] in dev_codes:
            by_code[row["class_code"]].append(row)
    selected: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for code in sorted(dev_codes):
        candidates = sorted(by_code[code], key=lambda row: deterministic_key("dev", code, row["sha256"]))
        if len(candidates) < 5:
            raise ValueError(f"{code} has fewer than five train candidates for dev")
        chosen = candidates[:5]
        if len({row["sha256"] for row in chosen}) != 5:
            raise ValueError(f"duplicate content selected for dev class {code}")
        selected.extend(chosen)
        hashes.update(row["sha256"] for row in chosen)
    return selected, hashes


def choose_dev_unknowns(
    unknown: set[str], domains: dict[str, str], train_counts: dict[str, int],
    recalls: dict[str, float | None],
) -> set[str]:
    """Freeze half of each domain's trainable hard-unknown classes for dev."""
    targets = {"disease": 36, "pest": 18}
    selected: set[str] = set()
    for domain, target in targets.items():
        candidates = [code for code in unknown if domains[code] == domain and train_counts[code] >= 5]
        ordered = sorted(
            candidates,
            key=lambda code: (
                0 if recalls[code] is not None else 1,
                -(1 - recalls[code]) if recalls[code] is not None else 0,
                train_counts[code], deterministic_key("dev-unknown", code),
            ),
        )
        if len(ordered) < target:
            raise ValueError(f"not enough trainable {domain} unknown classes for dev")
        selected.update(ordered[:target])
    return selected


def near_duplicate_audit(rows: list[dict[str, Any]], threshold: int, workers: int, cache_path: Path) -> list[dict[str, Any]]:
    """Return pHash candidate pairs crossing a training/evaluation boundary."""
    if threshold < 0 or threshold > 6:
        raise ValueError("pHash threshold must be in [0, 6] for the exact seven-segment index")
    if workers < 1:
        raise ValueError("pHash workers must be positive")
    cache = {
        str(row.get("sha256")): str(row.get("phash"))
        for row in read_jsonl(cache_path)
        if row.get("sha256") and row.get("phash")
    } if cache_path.is_file() else {}
    missing = [row for row in rows if row["sha256"] not in cache]
    print(f"pHash: cache hits={len(rows) - len(missing)}, computing={len(missing)} images with {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for row, value in zip(missing, executor.map(phash, (Path(item["source_path"]) for item in missing)), strict=True):
            cache[row["sha256"]] = value
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(cache_path, [{"sha256": key, "phash": cache[key]} for key in sorted(cache)])
    for row in rows:
        row["phash"] = cache[row["sha256"]]
    print("pHash: indexing cross-boundary candidate pairs", flush=True)
    evaluation = [row for row in rows if row["image_split"] in {"dev", "test", "milvus_reference"}]
    training = [row for row in rows if row["image_split"] == "train_candidate"]
    pairs: list[dict[str, Any]] = []
    # Split the 256-bit hash into seven disjoint segments. With <= 6 bit
    # differences, at least one segment must match exactly (pigeonhole
    # principle), so this index has no false negatives at the configured
    # threshold while avoiding 1,422 x ~194k exhaustive comparisons.
    segments = [(0, 37), (37, 74), (74, 111), (111, 148), (148, 185), (185, 222), (222, 256)]
    buckets: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for item in training:
        for index, (start, end) in enumerate(segments):
            buckets[index, item["phash"][start:end]].append(item)
    for item in evaluation:
        candidates = {
            candidate["sha256"]: candidate
            for index, (start, end) in enumerate(segments)
            for candidate in buckets.get((index, item["phash"][start:end]), [])
        }
        for candidate in candidates.values():
            distance = hamming(item["phash"], candidate["phash"])
            if 0 < distance <= threshold:
                pairs.append({
                    "evaluation_sha256": item["sha256"], "evaluation_split": item["image_split"],
                    "evaluation_path": item["source_path"], "evaluation_class": item["class_code"],
                    "training_sha256": candidate["sha256"], "training_path": candidate["source_path"],
                    "training_class": candidate["class_code"], "phash_distance": distance,
                    "resolution": "exclude_training_candidate_pending_review",
                })
    return sorted(pairs, key=lambda item: (item["evaluation_split"], item["evaluation_sha256"], item["phash_distance"]))


def public_eval_row(row: dict[str, Any], class_role: str, evaluation_bucket: str) -> dict[str, Any]:
    return {
        "id": f"{evaluation_bucket}-{row['sha256'][:20]}",
        "images": [row["v1_1_path"]], "image_sha256": row["sha256"],
        "task": "agricultural_image_classification", "class_role": class_role,
        "evaluation_bucket": evaluation_bucket,
        "metadata": {"domain": row["domain"], "source_split": row["image_split"]},
    }


def private_truth_row(row: dict[str, Any], class_role: str, evaluation_bucket: str) -> dict[str, Any]:
    return {
        "id": f"{evaluation_bucket}-{row['sha256'][:20]}", "image_sha256": row["sha256"],
        "class_code": row["class_code"], "class_role": class_role,
        "evaluation_bucket": evaluation_bucket, "domain": row["domain"],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    input_root = args.input_root.resolve()
    output_entry = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    output_root = output_entry.resolve()
    if output_root.exists():
        if not args.replace:
            raise ValueError(f"output already exists: {output_root}; use --replace after review")
        shutil.rmtree(output_root)
    manifests = input_root / "manifests"
    train_rows, train_duplicates = deduplicate_content(
        image_rows(manifests / "train.jsonl", "train_candidate"), "train_candidate"
    )
    test_rows, test_duplicates = deduplicate_content(
        image_rows(manifests / "test.jsonl", "test"), "test"
    )
    reference_rows, reference_duplicates = deduplicate_content(
        image_rows(manifests / "milvus_reference.jsonl", "milvus_reference"), "milvus_reference"
    )
    input_duplicates = train_duplicates + test_duplicates + reference_duplicates
    evaluation_hashes = {row["sha256"] for row in test_rows + reference_rows}
    train_exact_conflicts = [row for row in train_rows if row["sha256"] in evaluation_hashes]
    train_rows = [row for row in train_rows if row["sha256"] not in evaluation_hashes]
    input_duplicates.extend({
        "sha256": row["sha256"], "split": "train_candidate", "class_code": row["class_code"],
        "image_name": row["image_name"], "source_path": row["source_path"],
        "resolution": "excluded_train_exact_duplicate_of_evaluation",
    } for row in train_exact_conflicts)
    all_rows = train_rows + test_rows + reference_rows
    codes = {row["class_code"] for row in all_rows}
    domains = {row["class_code"]: row["domain"] for row in all_rows}
    if len(codes) != 217:
        raise ValueError(f"expected 217 classes, found {len(codes)}")
    train_counts = read_coverage(manifests / "class_coverage.csv")
    if set(train_counts) != codes:
        raise ValueError("class-coverage manifest does not match image manifests")
    similar = load_similar(args.similar_classes.resolve(), codes)
    recalls, raw_base_report = raw_base_recalls(args.raw_base_scored.resolve(), codes, args.allow_partial_raw_base)
    targets = {"disease": 72, "pest": 36}
    unknown, reasons = choose_unknowns(codes, domains, train_counts, recalls, similar, targets)
    known = codes - unknown
    if Counter(domains[code] for code in known) != Counter({"disease": 73, "pest": 36}):
        raise ValueError("known-class domain targets are not satisfied")
    dev_unknown = choose_dev_unknowns(unknown, domains, train_counts, recalls)
    dev_codes = known | dev_unknown
    dev_rows, dev_hashes = choose_dev_images(train_rows, dev_codes)
    for row in dev_rows:
        row["image_split"] = "dev"
    candidate_rows = [row for row in train_rows if row["sha256"] not in dev_hashes]
    exact = Counter(row["sha256"] for row in all_rows)
    if any(count > 1 for count in exact.values()):
        raise ValueError("v1 contains a test/Milvus SHA-256 overlap after per-split deduplication")
    audit_rows = candidate_rows + dev_rows + test_rows + reference_rows
    near_pairs = near_duplicate_audit(audit_rows, args.phash_threshold, args.phash_workers, args.phash_cache.resolve())
    near_train_hashes = {row["training_sha256"] for row in near_pairs}
    candidate_rows = [row for row in candidate_rows if row["sha256"] not in near_train_hashes]
    for row in candidate_rows:
        row["image_split"] = "train_candidate"
    materialized = candidate_rows + dev_rows + test_rows + reference_rows
    split_hashes: dict[str, set[str]] = defaultdict(set)
    for row in materialized:
        split_hashes[row["image_split"]].add(row["sha256"])
    for left in split_hashes:
        for right in split_hashes:
            if left < right and split_hashes[left] & split_hashes[right]:
                raise ValueError(f"output SHA-256 overlap: {left}/{right}")

    for row in materialized:
        destination = output_root / "images" / row["image_split"] / row["domain"] / row["class_code"] / row["image_name"]
        logical_destination = output_entry / "images" / row["image_split"] / row["domain"] / row["class_code"] / row["image_name"]
        if row["image_split"] == "train_candidate":
            relative_link(Path(row["source_path"]), destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["source_path"], destination)
        row["v1_1_path"] = str(logical_destination.relative_to(REPO_ROOT))

    class_rows: list[dict[str, Any]] = []
    for code in sorted(codes):
        role = "unknown" if code in unknown else "known"
        bridge = next((value for value in similar[code] if value in known), None) if role == "unknown" else None
        if role == "unknown" and not bridge:
            raise ValueError(f"unknown class {code} has no known bridge after selection")
        class_rows.append({
            "class_code": code, "domain": domains[code], "class_role": role,
            "dev_role": "unknown_dev" if code in dev_unknown else ("known_dev" if code in known else "unknown_holdout"),
            "train_candidate_images_before_near_dedup": train_counts[code],
            "raw_base_recall": recalls[code], "raw_base_missing": recalls[code] is None,
            "selection_reason": reasons.get(code, "complement_of_unknown_selection"),
            "milvus_top3_similar_classes": similar[code], "known_bridge_class": bridge,
            "sft_eligible": role == "known",
        })

    image_pool = []
    for row in materialized:
        role = "unknown" if row["class_code"] in unknown else "known"
        image_pool.append({
            "image_sha256": row["sha256"], "phash": row["phash"], "image_path": row["v1_1_path"],
            "source_path": row["source_path"], "class_code": row["class_code"], "domain": row["domain"],
            "image_split": row["image_split"], "class_role": role,
            "sft_eligible": row["image_split"] == "train_candidate" and role == "known",
            "dev_eligible": row["image_split"] == "dev", "milvus_eligible": row["image_split"] == "milvus_reference",
        })
    task_candidates = []
    for row in image_pool:
        if not row["sft_eligible"]:
            continue
        for route in ("direct", "hcv_rag"):
            for language in ("en", "zh"):
                for question_type in ("open", "option"):
                    task_candidates.append({
                        "candidate_id": hashlib.sha256(
                            f"{row['image_sha256']}:{route}:{language}:{question_type}".encode()
                        ).hexdigest()[:24],
                        "query_image_sha256": row["image_sha256"], "query_image": row["image_path"],
                        "class_code": row["class_code"], "route": route, "language": language,
                        "question_type": question_type, "split": "train", "requires_human_or_teacher_acceptance": True,
                    })
    eligibility = [{
        "class_code": row["class_code"], "class_role": row["class_role"], "sft_eligible": row["sft_eligible"],
        "dev_evaluation_eligible": row["dev_role"] in {"known_dev", "unknown_dev"},
    } for row in class_rows]

    dev_public = [public_eval_row(row, "unknown" if row["class_code"] in unknown else "known", "unknown_dev" if row["class_code"] in dev_unknown else "known") for row in dev_rows]
    dev_truth = [private_truth_row(row, "unknown" if row["class_code"] in unknown else "known", "unknown_dev" if row["class_code"] in dev_unknown else "known") for row in dev_rows]
    test_public = [public_eval_row(row, "unknown" if row["class_code"] in unknown else "known", "unknown_dev" if row["class_code"] in dev_unknown else ("unknown_holdout" if row["class_code"] in unknown else "known")) for row in test_rows]
    test_truth = [private_truth_row(row, "unknown" if row["class_code"] in unknown else "known", "unknown_dev" if row["class_code"] in dev_unknown else ("unknown_holdout" if row["class_code"] in unknown else "known")) for row in test_rows]
    duplicate_records = input_duplicates
    rejected = [{"sha256": value, "reason": "near_duplicate_of_evaluation_image", "resolution": "excluded_from_train_candidate"} for value in sorted(near_train_hashes)]

    write_jsonl(output_root / "manifests/class_split.jsonl", class_rows)
    write_json(output_root / "manifests/raw_base_coverage.json", raw_base_report)
    write_jsonl(output_root / "manifests/images.jsonl", image_pool)
    write_jsonl(output_root / "vlm_data/candidates/image_pool.jsonl", image_pool)
    write_jsonl(output_root / "vlm_data/candidates/task_candidates.jsonl", task_candidates)
    write_jsonl(output_root / "vlm_data/candidates/class_task_eligibility.jsonl", eligibility)
    write_jsonl(output_root / "vlm_data/accepted/train.jsonl", [])
    write_jsonl(output_root / "vlm_data/accepted/dev.jsonl", dev_public)
    write_jsonl(output_root / "vlm_data/accepted/test_public.jsonl", test_public)
    write_jsonl(output_root / "vlm_data/accepted/private/dev_truth.jsonl", dev_truth)
    write_jsonl(output_root / "vlm_data/accepted/private/test_truth.jsonl", test_truth)
    write_jsonl(output_root / "vlm_data/audits/duplicate_records.jsonl", duplicate_records)
    write_jsonl(output_root / "vlm_data/audits/image_split_conflicts.jsonl", [])
    write_jsonl(output_root / "vlm_data/audits/near_duplicate_clusters.jsonl", near_pairs)
    write_jsonl(output_root / "vlm_data/audits/rejected_candidates.jsonl", rejected)

    summary = {
        "schema_version": "agrinet.open-agri-v1.1-hcv-base/v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "input_root": str(input_root), "input_summary_sha256": digest(manifests / "summary.json"),
        "partial_raw_base": bool(raw_base_report["classes_missing"]), "raw_base": raw_base_report,
        "counts": {
            "classes": len(codes), "known_classes": len(known), "unknown_classes": len(unknown),
            "dev_classes": len(dev_codes), "dev_images": len(dev_rows), "test_images": len(test_rows),
            "milvus_reference_images": len(reference_rows), "train_candidate_images": len(candidate_rows),
            "task_candidates": len(task_candidates), "near_duplicate_training_exclusions": len(near_train_hashes),
            "exact_duplicate_training_exclusions": len(train_exact_conflicts),
        },
        "policy": {
            "unknown": "model-train-unseen but Milvus-covered; every unknown has a known Top-3 bridge",
            "dev": "all known classes plus half of trainable unknown classes, five images per class, evaluation only",
            "test": "frozen final test covering all classes; labels stored under vlm_data/accepted/private",
            "train": "known-class candidates only; accepted SFT rows are populated only after review",
            "near_duplicates": "evaluation-first; cross-boundary pHash candidates are excluded from training pending review",
        },
    }
    write_json(output_root / "manifests/summary.json", summary)
    write_json(output_root / "vlm_data/audits/acceptance_summary.json", {
        "accepted_train_rows": 0, "accepted_dev_rows": len(dev_public), "accepted_test_rows": len(test_public),
        "task_candidates": len(task_candidates), "dedup_policy": "canonical task fingerprint required before writing accepted/train.jsonl",
    })
    return summary


def main() -> int:
    args = parse_args()
    summary = build(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
