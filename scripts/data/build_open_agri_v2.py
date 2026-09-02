#!/usr/bin/env python3
"""Build the formal test-blind open_agri_v2 benchmark view.

Roles use only the v1 training manifest, a frozen class catalog, and a frozen
SigLIP2/Milvus neighbour graph. The final-test images, IDs, public inputs, and
private truth are not read until role selection is complete and are never
selection inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
V1_1_ROOT = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v1_1_hcv_base"
V1_ROOT = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v1"
DEFAULT_OUTPUT = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v2"
DEFAULT_SIMILAR = REPO_ROOT / "outputs/artifacts/datasets/m1-direct-current-hcv-v1/similar_classes.jsonl"
DEFAULT_CLASS_CATALOG = REPO_ROOT / "outputs/artifacts/datasets/m1-direct-current-hcv-v1/classes.jsonl"
DEFAULT_PHASH_CACHE = REPO_ROOT / "outputs/artifacts/datasets/open-agri-v2-formal-phash-cache-v1.jsonl"
V1_1_PHASH_CACHE = REPO_ROOT / "outputs/artifacts/datasets/open-agri-v2-phash-cache-v1.jsonl"
SEED = "open-agri-v2-formal-test-blind-20260903"
TAIL_FRACTION = 0.40
CONFUSABLE_FRACTION = 0.40
MIN_KNOWN_FINAL_TRAIN = 25
MIN_KNOWN_DEV = 3
MAX_KNOWN_DEV = 5
# Frozen exclusions discovered by the same pHash policy on the full v1 training
# source. Candidates here cannot be bridge anchors because the audit would
# leave them below the final Known training floor.
PHASH_INELIGIBLE_BRIDGE_ANCHORS = {"N04052"}
PHASH_INELIGIBLE_KNOWN = {"N04052"}
BUCKET_FRACTIONS = (
    ("tail_confusable", 0.40), ("tail_distinct", 0.25),
    ("head_confusable", 0.25), ("head_distinct", 0.10),
)

_spec = importlib.util.spec_from_file_location("open_agri_v1_1_builder", REPO_ROOT / "scripts/data/build_open_agri_v1_1_hcv_base.py")
assert _spec and _spec.loader
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

_historical_spec = importlib.util.spec_from_file_location(
    "open_agri_v2_historical_migration",
    REPO_ROOT / "scripts/data/migrate_open_agri_v1_1_hcv_base_historical_sft.py",
)
assert _historical_spec and _historical_spec.loader
historical_migration = importlib.util.module_from_spec(_historical_spec)
_historical_spec.loader.exec_module(historical_migration)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return v2.read_jsonl(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    v2.write_jsonl(path, rows)


def write_json(path: Path, value: Any) -> None:
    v2.write_json(path, value)


def digest(path: Path) -> str:
    return v2.digest(path)


def stable_key(*values: str) -> str:
    return hashlib.sha256((SEED + ":" + ":".join(values)).encode()).hexdigest()


def rounded_targets(total: int) -> dict[str, int]:
    """Largest-remainder quotas for the four declared Unknown buckets."""
    raw = [(name, total * fraction) for name, fraction in BUCKET_FRACTIONS]
    result = {name: int(value) for name, value in raw}
    for name, _ in sorted(raw, key=lambda item: (item[1] - int(item[1]), stable_key("quota", item[0])), reverse=True)[:total - sum(result.values())]:
        result[name] += 1
    return result


def profile_classes(train_counts: dict[str, int], domains: dict[str, str], similar: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    """Derive test-free long-tail and visual-confusability metadata."""
    reverse: dict[str, set[str]] = defaultdict(set)
    for code, neighbours in similar.items():
        for neighbour in neighbours:
            reverse[neighbour].add(code)
    profile: dict[str, dict[str, Any]] = {}
    for domain in ("disease", "pest"):
        codes = sorted(code for code, value in domains.items() if value == domain)
        tail_n = max(1, round(len(codes) * TAIL_FRACTION))
        confusable_n = max(1, round(len(codes) * CONFUSABLE_FRACTION))
        tail = set(sorted(codes, key=lambda code: (train_counts[code], stable_key("tail", code)))[:tail_n])
        reciprocal = {code: sum(code in similar[neighbour] for neighbour in similar[code]) for code in codes}
        confusable = set(sorted(codes, key=lambda code: (-reciprocal[code], -len(reverse[code]), stable_key("confusable", code)))[:confusable_n])
        for code in codes:
            bucket = ("tail_" if code in tail else "head_") + ("confusable" if code in confusable else "distinct")
            profile[code] = {
                "train_candidate_images_before_dev": train_counts[code],
                "long_tail": code in tail, "long_tail_fraction": TAIL_FRACTION,
                "visual_confusable": code in confusable, "visual_confusable_fraction": CONFUSABLE_FRACTION,
                "reciprocal_top3_neighbours": reciprocal[code], "incoming_top3_neighbours": len(reverse[code]),
                "stratification_bucket": bucket,
            }
    return profile


def select_unknowns(train_counts: dict[str, int], domains: dict[str, str], similar: dict[str, list[str]], targets: dict[str, int]) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """Use deterministic strata and a known-top3 bridge; never read final test."""
    codes = set(train_counts)
    profile = profile_classes(train_counts, domains, similar)
    unknown = {code for code in codes if train_counts[code] < 5}
    records = {
        code: {**profile[code], "selection_reason": "no_train_candidate" if train_counts[code] == 0 else "insufficient_train_candidates_for_known_dev", "selection_phase": "forced_capacity"}
        for code in unknown
    }

    def bridgeable(code: str) -> bool:
        proposed = unknown | {code}
        known = codes - proposed
        return all(any(neighbour in known for neighbour in similar[item]) for item in proposed)

    for domain, target in targets.items():
        if sum(domains[code] == domain for code in unknown) > target:
            raise ValueError(f"forced unknowns exceed {domain} target")
        quotas = rounded_targets(target)
        for bucket, _ in BUCKET_FRACTIONS:
            selected = sum(domains[code] == domain and profile[code]["stratification_bucket"] == bucket for code in unknown)
            needed = max(0, quotas[bucket] - selected)
            candidates = sorted((code for code in codes if domains[code] == domain and code not in unknown and profile[code]["stratification_bucket"] == bucket), key=lambda code: stable_key("bucket", bucket, code))
            for code in candidates:
                if needed == 0:
                    break
                if bridgeable(code):
                    unknown.add(code)
                    records[code] = {**profile[code], "selection_reason": "stratified_long_tail_visual_confusability", "selection_phase": "bucket"}
                    needed -= 1
        remaining = target - sum(domains[code] == domain for code in unknown)
        for bucket, _ in BUCKET_FRACTIONS:
            if remaining == 0:
                break
            candidates = sorted((code for code in codes if domains[code] == domain and code not in unknown and profile[code]["stratification_bucket"] == bucket), key=lambda code: stable_key("fallback", bucket, code))
            for code in candidates:
                if remaining == 0:
                    break
                if bridgeable(code):
                    unknown.add(code)
                    records[code] = {**profile[code], "selection_reason": "stratified_fallback_after_bucket_or_bridge_constraint", "selection_phase": "fallback"}
                    remaining -= 1
        if remaining:
            raise ValueError(f"could not select {target} bridgeable {domain} unknown classes")
    known = codes - unknown
    broken = [code for code in unknown if not any(value in known for value in similar[code])]
    if broken:
        raise ValueError(f"unknown classes without known Top-3 bridge: {broken}")
    return unknown, records


def select_dev_unknowns(unknown: set[str], class_rows: dict[str, dict[str, Any]]) -> set[str]:
    selected: set[str] = set()
    for domain in ("disease", "pest"):
        candidates = [code for code in unknown if class_rows[code]["domain"] == domain and int(class_rows[code]["train_candidate_images_before_dev"]) >= 5]
        target = (len(candidates) + 1) // 2
        selected.update(sorted(candidates, key=lambda code: stable_key("dev-unknown", code))[:target])
    return selected


def select_known_anchors(
    initial_known: set[str], train_counts: dict[str, int], domains: dict[str, str], similar: dict[str, list[str]],
) -> tuple[set[str], set[str], dict[str, dict[str, Any]]]:
    """Enforce final training coverage while preserving strict Top-3 bridges."""
    codes = set(train_counts)
    forced = {
        code for code in initial_known
        if train_counts[code] - MIN_KNOWN_DEV < MIN_KNOWN_FINAL_TRAIN
        or code in PHASH_INELIGIBLE_KNOWN
    }
    base_known = initial_known - forced
    initial_unknown = codes - base_known
    broken = {
        code for code in initial_unknown
        if not any(neighbour in base_known for neighbour in similar[code])
    }
    candidates = sorted({
        neighbour
        for code in broken
        for neighbour in similar[code]
        if neighbour in initial_unknown
        and neighbour not in forced
        and neighbour not in PHASH_INELIGIBLE_BRIDGE_ANCHORS
        and train_counts[neighbour] - MIN_KNOWN_DEV >= MIN_KNOWN_FINAL_TRAIN
    }, key=lambda code: stable_key("bridge-anchor", code))
    solutions: list[tuple[str, ...]] = []
    for size in range(len(candidates) + 1):
        for promoted in itertools.combinations(candidates, size):
            known = base_known | set(promoted)
            if all(any(neighbour in known for neighbour in similar[code]) for code in codes - known):
                solutions.append(promoted)
        if solutions:
            break
    if not solutions:
        raise ValueError("no capacity-qualified strict Top-3 bridge-anchor solution")

    def objective(promoted: tuple[str, ...]) -> tuple[Any, ...]:
        disease_delta = abs(sum(domains[code] == "disease" for code in promoted) - sum(domains[code] == "disease" for code in forced))
        pest_delta = abs(sum(domains[code] == "pest" for code in promoted) - sum(domains[code] == "pest" for code in forced))
        return (disease_delta + pest_delta, -sum(train_counts[code] for code in promoted), tuple(stable_key("anchor-tie", code) for code in promoted))

    promoted = set(min(solutions, key=objective))
    known = base_known | promoted
    if any(train_counts[code] - MIN_KNOWN_DEV < MIN_KNOWN_FINAL_TRAIN for code in known):
        raise ValueError("known final training capacity constraint violated")
    records: dict[str, dict[str, Any]] = {
        code: {"selection_reason": "forced_unknown_final_train_below_25", "selection_phase": "forced_final_train_capacity"}
        for code in forced
    }
    for code in promoted:
        records[code] = {"selection_reason": "minimal_strict_top3_bridge_anchor_promotion", "selection_phase": "bridge_anchor_promotion"}
    return known, forced, records


def choose_v2_dev_images(
    train_rows: list[dict[str, Any]], dev_codes: set[str], known: set[str], train_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Reserve 3--5 dev images while leaving every Known at least 25 train rows."""
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        if row["class_code"] in dev_codes:
            by_code[row["class_code"]].append(row)
    selected: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for code in sorted(dev_codes):
        if code in known:
            target = min(MAX_KNOWN_DEV, train_counts[code] - MIN_KNOWN_FINAL_TRAIN)
            if target < MIN_KNOWN_DEV:
                raise ValueError(f"{code} cannot satisfy Known dev/train capacity")
        else:
            target = 5
        candidates = sorted(by_code[code], key=lambda row: stable_key("dev", code, row["sha256"]))
        if len(candidates) < target:
            raise ValueError(f"{code} has fewer than {target} train candidates for dev")
        selected.extend(candidates[:target])
        hashes.update(row["sha256"] for row in candidates[:target])
    return selected, hashes


def copy_or_link(source: Path, destination: Path, *, linked: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if linked:
        destination.symlink_to(os.path.relpath(source, destination.parent))
    else:
        shutil.copy2(source, destination)


def v1_1_contract(v1_1_root: Path) -> dict[str, str]:
    return {key + "_sha256": digest(path) for key, path in {
        "test_public": v1_1_root / "vlm_data/accepted/test_public.jsonl",
        "test_truth": v1_1_root / "vlm_data/accepted/private/test_truth.jsonl",
        "images": v1_1_root / "manifests/images.jsonl",
    }.items()}


def migrate_historical_from_raw_sources(output: Path) -> dict[str, int]:
    """Recanonicalize every audited source against the final v2 image pool.

    Never inherit v1.1's historical Known/Unknown filter here: classes can
    change role between versions. The shared migration implementation performs
    format validation, source-priority deduplication, and final-v2
    Known/train-candidate isolation directly from its registered raw sources.
    """
    summary = historical_migration.migrate(output, replace=True)
    return {"direct": int(summary["direct"]), "rag": int(summary["rag"])}


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root.resolve()
    if output.exists():
        if not args.replace:
            raise ValueError(f"output exists: {output}; use --replace after review")
        shutil.rmtree(output)
    v1_1_root = args.v1_1_root.resolve()
    # Role selection occurs before opening any v2 file. This source manifest
    # contains training candidates only (no test images, IDs, or labels).
    role_source = args.v1_root.resolve() / "manifests/train.jsonl"
    catalog_source = args.class_catalog.resolve()
    catalog = read_jsonl(catalog_source)
    domains = {str(row["code"]): str(row["task_domain"]) for row in catalog}
    train_counts = Counter(str(row["class_code"]) for row in read_jsonl(role_source))
    train_counts = {code: int(train_counts.get(code, 0)) for code in domains}
    if len(train_counts) != 217:
        raise ValueError("training coverage must contain 217 classes")
    similar = v2.load_similar(args.similar_classes.resolve(), set(train_counts))
    initial_unknown, selection = select_unknowns(train_counts, domains, similar, {"disease": 72, "pest": 36})
    initial_known = set(train_counts) - initial_unknown
    known, forced_unknown, anchor_selection = select_known_anchors(initial_known, train_counts, domains, similar)
    unknown = set(train_counts) - known
    selection.update(anchor_selection)
    class_rows_by_code = {code: {"class_code": code, "domain": domains[code], **profile} for code, profile in profile_classes(train_counts, domains, similar).items()}
    for code, record in selection.items():
        class_rows_by_code[code].update(record)
    dev_unknown = select_dev_unknowns(unknown, class_rows_by_code)
    dev_codes = known | dev_unknown

    v1_1_images = read_jsonl(v1_1_root / "manifests/images.jsonl")
    evaluation_hashes = {
        str(row["image_sha256"])
        for row in v1_1_images
        if row["image_split"] in {"test", "milvus_reference"}
    }
    # Candidate availability is defined from the training-only v1 source, not
    # from v1.1's historical role-filtered pool. This preserves the test-blind
    # selection contract and admits valid former-Unknown bridge anchors.
    candidate_source = [
        {
            "class_code": str(row["class_code"]), "domain": str(row["domain"]),
            "image_name": str(row["image_name"]), "source_path": str(row["source_path"]),
            "sha256": digest(Path(str(row["source_path"]))),
        }
        for row in read_jsonl(args.v1_root.resolve() / "manifests/train.jsonl")
    ]
    candidate_source = [row for row in candidate_source if row["sha256"] not in evaluation_hashes]
    all_train = list({row["sha256"]: row for row in candidate_source}.values())
    for row in all_train:
        row["image_split"] = "train_candidate"
    actual_counts = Counter(row["class_code"] for row in all_train)
    if any(actual_counts[code] - MIN_KNOWN_DEV < MIN_KNOWN_FINAL_TRAIN for code in known):
        raise ValueError("known class cannot supply minimum formal-v2 dev and training capacity")
    dev_rows, dev_hashes = choose_v2_dev_images(all_train, dev_codes, known, actual_counts)
    for row in dev_rows:
        row["image_split"] = "dev"
    candidate_rows = [row for row in all_train if row["sha256"] not in dev_hashes and row["class_code"] in known]
    test_rows = [dict(row, image_name=Path(str(row["source_path"])).name, sha256=str(row["image_sha256"])) for row in v1_1_images if row["image_split"] == "test"]
    reference_rows = [dict(row, image_name=Path(str(row["source_path"])).name, sha256=str(row["image_sha256"])) for row in v1_1_images if row["image_split"] == "milvus_reference"]
    if len(test_rows) != 1019 or len(reference_rows) != 401:
        raise ValueError("v2 immutable test/reference inventory mismatch")
    all_rows = candidate_rows + dev_rows + test_rows + reference_rows
    cache = args.phash_cache.resolve()
    if not cache.exists() and V1_1_PHASH_CACHE.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(V1_1_PHASH_CACHE, cache)
    near_pairs = v2.near_duplicate_audit(all_rows, args.phash_threshold, args.phash_workers, cache)
    excluded_near = {row["training_sha256"] for row in near_pairs}
    candidate_rows = [row for row in candidate_rows if row["sha256"] not in excluded_near]
    for row in candidate_rows:
        row["image_split"] = "train_candidate"
    final_train_counts = Counter(row["class_code"] for row in candidate_rows)
    underfilled = {code: final_train_counts[code] for code in known if final_train_counts[code] < MIN_KNOWN_FINAL_TRAIN}
    if underfilled:
        raise ValueError(f"pHash audit left Known classes below {MIN_KNOWN_FINAL_TRAIN} final train images: {underfilled}")
    materialized = candidate_rows + dev_rows + test_rows + reference_rows
    split_hashes: dict[str, set[str]] = defaultdict(set)
    for row in materialized:
        split_hashes[row["image_split"]].add(row["sha256"])
    if any(split_hashes[left] & split_hashes[right] for left in split_hashes for right in split_hashes if left < right):
        raise ValueError("formal-v2 SHA-256 overlap across materialized splits")
    for row in materialized:
        split = row["image_split"]
        destination = output / "images" / split / row["domain"] / row["class_code"] / row["image_name"]
        copy_or_link(Path(row["source_path"]), destination, linked=split in {"train_candidate", "dev"})
        row["v2_path"] = str(Path("datasets/AgriNet-1K/open_agri_v2") / "images" / split / row["domain"] / row["class_code"] / row["image_name"])
        row["v1_1_path"] = row["v2_path"]

    class_rows: list[dict[str, Any]] = []
    for code in sorted(train_counts):
        role = "unknown" if code in unknown else "known"
        bridge = next((value for value in similar[code] if value in known), None) if role == "unknown" else None
        if role == "unknown" and not bridge:
            raise ValueError(f"unknown {code} lacks a known Top-3 bridge")
        class_rows.append({
            **class_rows_by_code[code], "class_role": role,
            "dev_role": "unknown_dev" if code in dev_unknown else ("known_dev" if role == "known" else "unknown_holdout"),
            "selection_reason": class_rows_by_code[code].get("selection_reason", "complement_of_stratified_unknown_selection"),
            "selection_phase": class_rows_by_code[code].get("selection_phase", "known_complement"),
            "milvus_top3_similar_classes": similar[code], "known_bridge_class": bridge,
            "sft_eligible": role == "known", "test_informed_protocol": False,
            "initial_stratified_role": "unknown" if code in initial_unknown else "known",
            "forced_unknown_final_train_below_25": code in forced_unknown,
            "bridge_anchor_promotion": code in anchor_selection and anchor_selection[code]["selection_phase"] == "bridge_anchor_promotion",
        })
    pool: list[dict[str, Any]] = []
    for row in materialized:
        role = "unknown" if row["class_code"] in unknown else "known"
        pool.append({
            "image_sha256": row["sha256"], "phash": row["phash"], "image_path": row["v2_path"], "source_path": row["source_path"],
            "class_code": row["class_code"], "domain": row["domain"], "image_split": row["image_split"], "class_role": role,
            "sft_eligible": row["image_split"] == "train_candidate" and role == "known",
            "dev_eligible": row["image_split"] == "dev", "milvus_eligible": row["image_split"] == "milvus_reference",
        })
    def task_rows() -> Iterable[dict[str, Any]]:
        for row in pool:
            if not row["sft_eligible"]:
                continue
            for route in ("direct", "hcv_rag"):
                for language in ("en", "zh"):
                    for question_type in ("open", "option"):
                        candidate = f"{row['image_sha256']}:{route}:{language}:{question_type}"
                        yield {
                            "candidate_id": hashlib.sha256(candidate.encode()).hexdigest()[:24],
                            "query_image_sha256": row["image_sha256"], "query_image": row["image_path"], "class_code": row["class_code"],
                            "route": route, "language": language, "question_type": question_type, "split": "train",
                            "requires_human_or_teacher_acceptance": True,
                        }

    private_test = read_jsonl(v1_1_root / "vlm_data/accepted/private/test_truth.jsonl")
    private_by_id = {str(row["id"]): row for row in private_test}
    v1_1_public = read_jsonl(v1_1_root / "vlm_data/accepted/test_public.jsonl")
    image_by_sha = {str(row["image_sha256"]): row for row in pool}
    test_public, immutable_rows = [], []
    for item in v1_1_public:
        hidden = private_by_id[str(item["id"])]
        image = image_by_sha[str(item["image_sha256"])]
        role = "unknown" if hidden["class_code"] in unknown else "known"
        bucket = "unknown_dev" if hidden["class_code"] in dev_unknown else ("unknown_holdout" if role == "unknown" else "known")
        test_public.append({**item, "images": [image["image_path"]], "class_role": role, "evaluation_bucket": bucket})
        v1_1_image = next(row for row in v1_1_images if row["image_sha256"] == item["image_sha256"])
        immutable_rows.append({
            "id": item["id"], "image_sha256": item["image_sha256"], "v1_1_path": v1_1_image["image_path"], "v2_path": image["image_path"],
            "v1_1_file_sha256": digest(REPO_ROOT / v1_1_image["image_path"]), "v2_file_sha256": digest(REPO_ROOT / image["image_path"]),
            "private_truth_id": hidden["id"], "private_truth_image_sha256": hidden["image_sha256"],
        })
    target = output / "vlm_data/accepted/private/test_truth.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(v1_1_root / "vlm_data/accepted/private/test_truth.jsonl", target)
    reference_reuse = []
    for source in (row for row in v1_1_images if row["image_split"] == "milvus_reference"):
        image = image_by_sha[str(source["image_sha256"])]
        reference_reuse.append({
            "image_sha256": source["image_sha256"], "v1_1_path": source["image_path"], "v2_path": image["image_path"],
            "v1_1_file_sha256": digest(REPO_ROOT / source["image_path"]), "v2_file_sha256": digest(REPO_ROOT / image["image_path"]),
        })
    if not all(row["v1_1_file_sha256"] == row["v2_file_sha256"] == row["image_sha256"] for row in reference_reuse):
        raise ValueError("formal v2 Milvus reference content differs from v1.1")
    dev_public = [v2.public_eval_row(row, "unknown" if row["class_code"] in unknown else "known", "unknown_dev" if row["class_code"] in dev_unknown else "known") for row in dev_rows]
    dev_truth = [v2.private_truth_row(row, "unknown" if row["class_code"] in unknown else "known", "unknown_dev" if row["class_code"] in dev_unknown else "known") for row in dev_rows]
    eligibility = [{"class_code": row["class_code"], "class_role": row["class_role"], "sft_eligible": row["sft_eligible"], "dev_evaluation_eligible": row["dev_role"] in {"known_dev", "unknown_dev"}} for row in class_rows]
    write_jsonl(output / "manifests/class_split.jsonl", class_rows)
    write_jsonl(output / "manifests/images.jsonl", pool)
    write_jsonl(output / "vlm_data/candidates/image_pool.jsonl", pool)
    historical = migrate_historical_from_raw_sources(output)
    write_jsonl(output / "vlm_data/candidates/task_candidates.jsonl", task_rows())
    write_jsonl(output / "vlm_data/candidates/class_task_eligibility.jsonl", eligibility)
    accepted_train_root = output / "vlm_data/accepted/train"
    accepted_train_root.mkdir(parents=True, exist_ok=True)
    for domain in ("disease", "pest"):
        for route in ("direct", "rag"):
            write_jsonl(accepted_train_root / f"{domain}_{route}.jsonl", [])
    write_json(
        accepted_train_root / "summary.json",
        {
            "schema_version": "agrinet.open-agri-v2.accepted-train-views/v1",
            "partition": ["route", "task_domain"],
            "total_rows": 0,
            "views": {},
        },
    )
    write_jsonl(output / "vlm_data/accepted/dev.jsonl", dev_public)
    write_jsonl(output / "vlm_data/accepted/test_public.jsonl", test_public)
    write_jsonl(output / "vlm_data/accepted/private/dev_truth.jsonl", dev_truth)
    write_jsonl(output / "vlm_data/audits/near_duplicate_clusters.jsonl", near_pairs)
    write_jsonl(output / "vlm_data/audits/rejected_candidates.jsonl", [{"sha256": value, "reason": "near_duplicate_of_evaluation_image", "resolution": "excluded_from_train_candidate"} for value in sorted(excluded_near)])
    write_jsonl(output / "vlm_data/audits/image_split_conflicts.jsonl", [])
    write_jsonl(output / "vlm_data/audits/v1_1_test_immutable_reuse.jsonl", immutable_rows)
    write_jsonl(output / "vlm_data/audits/v1_1_milvus_reference_immutable_reuse.jsonl", reference_reuse)
    contract = v1_1_contract(v1_1_root)
    summary = {
        "schema_version": "agrinet.open-agri-v2-test-blind-stratified/v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "test_informed_protocol": False,
        "role_selection_inputs": {"training_manifest": "datasets/AgriNet-1K/open_agri_v1/manifests/train.jsonl", "training_manifest_sha256": digest(role_source), "class_catalog": "outputs/artifacts/datasets/m1-direct-current-hcv-v1/classes.jsonl", "class_catalog_sha256": digest(catalog_source), "similar_classes_sha256": digest(args.similar_classes.resolve()), "final_test_accessed_for_selection": False},
        "v1_1_immutable_contract": {**contract, "test_items_verified": len(immutable_rows), "milvus_reference_items_verified": len(reference_reuse), "all_test_image_sha256_match": all(row["v1_1_file_sha256"] == row["v2_file_sha256"] == row["image_sha256"] for row in immutable_rows), "all_milvus_reference_image_sha256_match": True},
        "counts": {"classes": 217, "known_classes": len(known), "unknown_classes": len(unknown), "forced_unknown_final_train_below_25": len(forced_unknown), "bridge_anchor_promotions": sum(record["selection_phase"] == "bridge_anchor_promotion" for record in anchor_selection.values()), "dev_classes": len(dev_codes), "dev_images": len(dev_rows), "test_images": len(test_rows), "milvus_reference_images": len(reference_rows), "train_candidate_images": len(candidate_rows), "task_candidates": sum(bool(row["sft_eligible"]) for row in pool) * 8, "near_duplicate_training_exclusions": len(excluded_near), "historical_canonical": historical},
        "historical_supervision": {
            "selection_protocol": "direct raw-source canonicalization against the final dataset image pool; do not inherit a prior version's Known/Unknown-filtered canonical view",
            "accepted": sum(historical.values()), "direct": historical["direct"], "rag": historical["rag"],
            "historical_summary": "vlm_data/historical/summary.json",
        },
        "policy": {"role_selection": "test-blind stratified holdout, then joint capacity/bridge repair: final Known train-candidate count is at least 25 after 3--5 dev reservations and pHash exclusion; classes below the threshold are forced Unknown; a minimum number of capacity-qualified former Unknowns are promoted as strict frozen SigLIP2/Milvus Top-3 bridge anchors", "test": "v2 public test and private audit truth copied unchanged; test images, IDs, public inputs, and truth were not used for role selection", "train": "only final known train-candidate images are SFT eligible", "dev": "all Known classes receive 3--5 images while retaining at least 25 final train candidates; half of trainable Unknown classes per domain receive five images", "reference": "v2 Milvus reference content copied unchanged"},
    }
    write_json(output / "manifests/summary.json", summary)
    write_json(output / "vlm_data/audits/v1_1_immutable_contract.json", contract)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-root", type=Path, default=V1_ROOT)
    parser.add_argument("--v1-1-root", type=Path, default=V1_1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--similar-classes", type=Path, default=DEFAULT_SIMILAR)
    parser.add_argument("--class-catalog", type=Path, default=DEFAULT_CLASS_CATALOG)
    parser.add_argument("--phash-cache", type=Path, default=DEFAULT_PHASH_CACHE)
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--phash-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, sort_keys=True))
