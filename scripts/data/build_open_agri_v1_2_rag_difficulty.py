#!/usr/bin/env python3
"""Build the test-informed open_agri_v1.2 RAG-difficulty intermediate view.

v2 remains immutable. v3 reuses its frozen test, private truth, and Milvus
reference content, but derives class roles, dev, and SFT eligibility solely
from the separately recorded raw-base RAG evidence artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
DEFAULT_OUTPUT = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v1_2_rag_difficulty"
DEFAULT_EVIDENCE = REPO_ROOT / "outputs/artifacts/datasets/open-agri-v3-rag-role-evidence-v1"
DEFAULT_SIMILAR = REPO_ROOT / "outputs/artifacts/datasets/m1-direct-current-hcv-v1/similar_classes.jsonl"
DEFAULT_PHASH_CACHE = REPO_ROOT / "outputs/artifacts/datasets/open-agri-v1.2-rag-difficulty-phash-cache-v1.jsonl"
V1_1_PHASH_CACHE = REPO_ROOT / "outputs/artifacts/datasets/open-agri-v2-phash-cache-v1.jsonl"
SEED = "open-agri-v1.2-rag-difficulty-20260902"

_spec = importlib.util.spec_from_file_location("open_agri_v1_1_builder", REPO_ROOT / "scripts/data/build_open_agri_v1_1_hcv_base.py")
assert _spec and _spec.loader
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return v2.read_jsonl(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    v2.write_jsonl(path, rows)


def write_json(path: Path, value: Any) -> None:
    v2.write_json(path, value)


def digest(path: Path) -> str:
    return v2.digest(path)


def stable_sha_key(value: str) -> str:
    return hashlib.sha256((SEED + ":" + value).encode()).hexdigest()


def select_unknowns(
    classes: list[dict[str, Any]], similar: dict[str, list[str]], targets: dict[str, int],
) -> tuple[set[str], dict[str, str]]:
    """Apply the declared conservative RAG-first ranking with a bridge guard."""
    by_code = {str(row["class_code"]): row for row in classes}
    codes = set(by_code)
    domains = {code: str(row["domain"]) for code, row in by_code.items()}
    unknown = {code for code, row in by_code.items() if int(row["train_candidate_images"]) < 5}
    reasons = {
        code: "no_train_candidate" if int(by_code[code]["train_candidate_images"]) == 0 else "insufficient_train_candidates_for_known_dev"
        for code in unknown
    }
    for domain, target in targets.items():
        pool = [code for code in codes if domains[code] == domain and code not in unknown]
        needed = target - sum(domains[code] == domain for code in unknown)
        if needed < 0:
            raise ValueError(f"forced unknowns exceed {domain} target")

        def ranking(code: str) -> tuple[Any, ...]:
            row = by_code[code]
            incomplete = row.get("evidence_status") != "complete"
            accuracy = float(row["rag_accuracy"]) if row.get("rag_accuracy") is not None else -1.0
            lower = float(row.get("wilson_lower_95") or 0.0)
            return (0 if incomplete else 1, accuracy, lower, int(row["train_candidate_images"]), stable_sha_key(code))

        selected: list[str] = []
        for code in sorted(pool, key=ranking):
            if len(selected) == needed:
                break
            candidate_unknown = unknown | set(selected) | {code}
            known = codes - candidate_unknown
            if all(any(neighbor in known for neighbor in similar[item]) for item in candidate_unknown):
                selected.append(code)
        if len(selected) != needed:
            raise ValueError(f"could not select {needed} bridgeable RAG-ranked unknown classes for {domain}")
        unknown.update(selected)
        for code in selected:
            reasons[code] = "evidence_incomplete_conservative_unknown" if by_code[code].get("evidence_status") != "complete" else "low_rag_accuracy_then_wilson_then_train_capacity"
    known = codes - unknown
    broken = [code for code in unknown if not any(value in known for value in similar[code])]
    if broken:
        raise ValueError(f"unknown classes without known Top-3 bridge: {broken}")
    return unknown, reasons


def select_dev_unknowns(unknown: set[str], class_rows: dict[str, dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for domain in ("disease", "pest"):
        pool = [code for code in unknown if class_rows[code]["domain"] == domain and int(class_rows[code]["train_candidate_images"]) >= 5]
        target = (len(pool) + 1) // 2
        ordered = sorted(pool, key=lambda code: (
            0 if class_rows[code].get("evidence_status") != "complete" else 1,
            float(class_rows[code]["rag_accuracy"]) if class_rows[code].get("rag_accuracy") is not None else -1.0,
            float(class_rows[code].get("wilson_lower_95") or 0.0),
            int(class_rows[code]["train_candidate_images"]), stable_sha_key("dev-unknown:" + code),
        ))
        result.update(ordered[:target])
    return result


def copy_or_link(source: Path, destination: Path, *, linked: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if linked:
        destination.symlink_to(os.path.relpath(source, destination.parent))
    else:
        shutil.copy2(source, destination)


def v2_test_reference_contract(v2_root: Path) -> dict[str, Any]:
    paths = {
        "test_public": v2_root / "vlm_data/accepted/test_public.jsonl",
        "test_truth": v2_root / "vlm_data/accepted/private/test_truth.jsonl",
        "images": v2_root / "manifests/images.jsonl",
    }
    return {key + "_sha256": digest(path) for key, path in paths.items()}


def migrate_historical(v2_root: Path, output_root: Path, image_pool: dict[str, dict[str, Any]]) -> dict[str, int]:
    source = v2_root / "vlm_data/historical/canonical"
    destination = output_root / "vlm_data/historical/canonical"
    audit: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for route in ("direct", "rag"):
        kept: list[dict[str, Any]] = []
        for row in read_jsonl(source / f"{route}.jsonl"):
            metadata = row.get("metadata") or {}
            image_sha = str(metadata.get("v2_image_sha256") or "")
            image = image_pool.get(image_sha)
            if image is None:
                audit.append({"sample_id": row.get("sample_id"), "route": route, "status": "excluded", "reason": "not_v3_known_train_candidate", "image_sha256": image_sha})
                continue
            item = json.loads(json.dumps(row, ensure_ascii=False))
            item["images"] = [image["image_path"]]
            item["metadata"]["class_role"] = "known"
            item["metadata"]["v3_image_sha256"] = image_sha
            item["metadata"]["v3_image_split"] = "train_candidate"
            fingerprint = hashlib.sha256(json.dumps({"v2": metadata.get("open_agri_v2_supervision_fingerprint"), "image": image["image_path"], "route": route}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            item["metadata"]["open_agri_v3_supervision_fingerprint"] = fingerprint
            kept.append(item)
            audit.append({"sample_id": row.get("sample_id"), "route": route, "status": "accepted", "reason": "known_train_candidate", "image_sha256": image_sha, "v3_fingerprint": fingerprint})
        write_jsonl(destination / f"{route}.jsonl", kept)
        counts[route] = len(kept)
    write_jsonl(output_root / "vlm_data/audits/historical_v2_boundary_exclusions.jsonl", audit)
    return dict(counts)


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root.resolve()
    if output.exists():
        if not args.replace:
            raise ValueError(f"output exists: {output}; use --replace after review")
        shutil.rmtree(output)
    evidence = args.evidence.resolve()
    evidence_classes = read_jsonl(evidence / "class_difficulty.jsonl")
    if len(evidence_classes) != 217:
        raise ValueError("class_difficulty.jsonl must contain all 217 classes")
    by_class = {str(row["class_code"]): row for row in evidence_classes}
    similar = v2.load_similar(args.similar_classes.resolve(), set(by_class))
    unknown, reasons = select_unknowns(evidence_classes, similar, {"disease": 72, "pest": 36})
    known = set(by_class) - unknown
    if Counter(by_class[code]["domain"] for code in known) != Counter({"disease": 73, "pest": 36}):
        raise ValueError("v3 known class quotas are invalid")
    dev_unknown = select_dev_unknowns(unknown, by_class)
    dev_codes = known | dev_unknown

    v2_root = args.v2_root.resolve()
    v1_root = args.v1_root.resolve()
    v2_images = read_jsonl(v2_root / "manifests/images.jsonl")
    # v2 candidate rows have already passed exact and pHash conflict filtering;
    # rebuild dev from their original public source files under the new role set.
    candidate_source = [
        dict(row, image_name=Path(str(row["source_path"])).name, sha256=str(row["image_sha256"]))
        for row in v2_images if row["image_split"] in {"train_candidate", "dev"}
    ]
    source_by_sha = {str(row["image_sha256"]): row for row in candidate_source}
    all_train = list(source_by_sha.values())
    for row in all_train:
        row["image_split"] = "train_candidate"
    train_counts = Counter(row["class_code"] for row in all_train)
    # Evidence must use the same actual candidate capacity used for v3 dev.
    if any(train_counts[code] < 5 and code not in unknown for code in known):
        raise ValueError("known class cannot supply five v3 dev images")
    dev_rows, dev_hashes = v2.choose_dev_images(all_train, dev_codes)
    for row in dev_rows:
        row["image_split"] = "dev"
    candidate_rows = [row for row in all_train if row["sha256"] not in dev_hashes and row["class_code"] in known]

    test_rows = [dict(row, image_name=Path(str(row["source_path"])).name, sha256=str(row["image_sha256"])) for row in v2_images if row["image_split"] == "test"]
    reference_rows = [dict(row, image_name=Path(str(row["source_path"])).name, sha256=str(row["image_sha256"])) for row in v2_images if row["image_split"] == "milvus_reference"]
    if len(test_rows) != 1019 or len(reference_rows) != 401:
        raise ValueError("v2 immutable test/reference inventory mismatch")
    # Enforce content isolation again with v3's new candidate/dev boundary.
    all_rows = candidate_rows + dev_rows + test_rows + reference_rows
    cache = args.phash_cache.resolve()
    if not cache.exists() and V2_PHASH_CACHE.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(V2_PHASH_CACHE, cache)
    near_pairs = v2.near_duplicate_audit(all_rows, args.phash_threshold, args.phash_workers, cache)
    excluded_near = {row["training_sha256"] for row in near_pairs}
    candidate_rows = [row for row in candidate_rows if row["sha256"] not in excluded_near]
    for row in candidate_rows:
        row["image_split"] = "train_candidate"
    materialized = candidate_rows + dev_rows + test_rows + reference_rows
    split_hashes: dict[str, set[str]] = defaultdict(set)
    for row in materialized:
        split_hashes[row["image_split"]].add(row["sha256"])
    for left in split_hashes:
        for right in split_hashes:
            if left < right and split_hashes[left] & split_hashes[right]:
                raise ValueError(f"v3 SHA-256 overlap: {left}/{right}")

    for row in materialized:
        code, domain, split = row["class_code"], row["domain"], row["image_split"]
        destination = output / "images" / split / domain / code / row["image_name"]
        source = Path(row["source_path"])
        # Test/reference are copied, matching the v2 immutable materialization;
        # train/dev remain links to audited original content.
        copy_or_link(source, destination, linked=split in {"train_candidate", "dev"})
        # ``datasets`` is a shared-root symlink in this checkout. Do not call
        # Path.resolve() here: it would turn the public logical repository
        # path into the shared backing path and corrupt manifest portability.
        row["v3_path"] = str(Path("datasets/AgriNet-1K/open_agri_v3") / "images" / split / domain / code / row["image_name"])
        # Compatibility alias for the v2 public-evaluation row helper; the
        # persisted v3 manifests use only the explicit v3_path field.
        row["v2_path"] = row["v3_path"]

    class_rows: list[dict[str, Any]] = []
    for code in sorted(by_class):
        evidence_row = by_class[code]
        role = "unknown" if code in unknown else "known"
        bridge = next((value for value in similar[code] if value in known), None) if role == "unknown" else None
        if role == "unknown" and not bridge:
            raise ValueError(f"unknown {code} lacks a known Milvus/SigLIP2 Top-3 bridge")
        class_rows.append({
            **evidence_row, "class_role": role,
            "dev_role": "unknown_dev" if code in dev_unknown else ("known_dev" if code in known else "unknown_holdout"),
            "selection_reason": reasons.get(code, "complement_of_unknown_selection"),
            "milvus_top3_similar_classes": similar[code], "known_bridge_class": bridge,
            "sft_eligible": role == "known", "test_informed_protocol": True,
        })
    pool: list[dict[str, Any]] = []
    for row in materialized:
        role = "unknown" if row["class_code"] in unknown else "known"
        pool.append({
            "image_sha256": row["sha256"], "phash": row["phash"], "image_path": row["v3_path"], "source_path": row["source_path"],
            "class_code": row["class_code"], "domain": row["domain"], "image_split": row["image_split"], "class_role": role,
            "sft_eligible": row["image_split"] == "train_candidate" and role == "known",
            "dev_eligible": row["image_split"] == "dev", "milvus_eligible": row["image_split"] == "milvus_reference",
        })
    pool_by_sha = {str(row["image_sha256"]): row for row in pool if row["image_split"] == "train_candidate" and row["sft_eligible"]}
    historical_counts = migrate_historical(v2_root, output, pool_by_sha)
    def task_rows() -> Iterable[dict[str, Any]]:
        for row in pool:
            if not row["sft_eligible"]:
                continue
            for route in ("direct", "hcv_rag"):
                for language in ("en", "zh"):
                    for question_type in ("open", "option"):
                        yield {
                            "candidate_id": hashlib.sha256(f"{row['image_sha256']}:{route}:{language}:{question_type}".encode()).hexdigest()[:24],
                            "query_image_sha256": row["image_sha256"], "query_image": row["image_path"], "class_code": row["class_code"],
                            "route": route, "language": language, "question_type": question_type, "split": "train", "requires_human_or_teacher_acceptance": True,
                        }
    task_count = sum(bool(row["sft_eligible"]) for row in pool) * 8
    eligibility = [{"class_code": row["class_code"], "class_role": row["class_role"], "sft_eligible": row["sft_eligible"], "dev_evaluation_eligible": row["dev_role"] in {"known_dev", "unknown_dev"}} for row in class_rows]
    # Preserve private human-audit truth byte-for-byte. The public v3 view
    # only rewrites its logical image paths and refreshed role metadata.
    private_test = read_jsonl(v2_root / "vlm_data/accepted/private/test_truth.jsonl")
    private_by_id = {str(row["id"]): row for row in private_test}
    v2_public = read_jsonl(v2_root / "vlm_data/accepted/test_public.jsonl")
    v3_by_sha = {str(row["image_sha256"]): row for row in pool}
    test_public = []
    immutable_rows = []
    for item in v2_public:
        hidden = private_by_id[str(item["id"])]
        image = v3_by_sha[str(item["image_sha256"])]
        role = "unknown" if hidden["class_code"] in unknown else "known"
        bucket = "unknown_dev" if hidden["class_code"] in dev_unknown else ("unknown_holdout" if role == "unknown" else "known")
        test_public.append({**item, "images": [image["image_path"]], "class_role": role, "evaluation_bucket": bucket})
        v2_image = next(row for row in v2_images if row["image_sha256"] == item["image_sha256"])
        actual_v3 = REPO_ROOT / image["image_path"]
        immutable_rows.append({
            "id": item["id"], "image_sha256": item["image_sha256"], "v2_path": v2_image["image_path"], "v3_path": image["image_path"],
            "v2_file_sha256": digest(REPO_ROOT / v2_image["image_path"]), "v3_file_sha256": digest(actual_v3),
            "private_truth_id": hidden["id"], "private_truth_image_sha256": hidden["image_sha256"],
        })
    target = output / "vlm_data/accepted/private/test_truth.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(v2_root / "vlm_data/accepted/private/test_truth.jsonl", target)
    reference_reuse = []
    for source in (row for row in v2_images if row["image_split"] == "milvus_reference"):
        image = v3_by_sha[str(source["image_sha256"])]
        reference_reuse.append({
            "image_sha256": source["image_sha256"], "v2_path": source["image_path"], "v3_path": image["image_path"],
            "v2_file_sha256": digest(REPO_ROOT / source["image_path"]), "v3_file_sha256": digest(REPO_ROOT / image["image_path"]),
        })
    if not all(row["v2_file_sha256"] == row["v3_file_sha256"] == row["image_sha256"] for row in reference_reuse):
        raise ValueError("v3 Milvus reference content differs from v2")
    dev_public = [v2.public_eval_row(row, "unknown" if row["class_code"] in unknown else "known", "unknown_dev" if row["class_code"] in dev_unknown else "known") for row in dev_rows]
    dev_truth = [v2.private_truth_row(row, "unknown" if row["class_code"] in unknown else "known", "unknown_dev" if row["class_code"] in dev_unknown else "known") for row in dev_rows]
    write_jsonl(output / "manifests/class_split.jsonl", class_rows)
    write_jsonl(output / "manifests/images.jsonl", pool)
    write_jsonl(output / "vlm_data/candidates/image_pool.jsonl", pool)
    write_jsonl(output / "vlm_data/candidates/task_candidates.jsonl", task_rows())
    write_jsonl(output / "vlm_data/candidates/class_task_eligibility.jsonl", eligibility)
    write_jsonl(output / "vlm_data/accepted/train.jsonl", [])
    write_jsonl(output / "vlm_data/accepted/dev.jsonl", dev_public)
    write_jsonl(output / "vlm_data/accepted/test_public.jsonl", test_public)
    write_jsonl(output / "vlm_data/accepted/private/dev_truth.jsonl", dev_truth)
    write_jsonl(output / "vlm_data/audits/near_duplicate_clusters.jsonl", near_pairs)
    write_jsonl(output / "vlm_data/audits/rejected_candidates.jsonl", [{"sha256": value, "reason": "near_duplicate_of_evaluation_image", "resolution": "excluded_from_train_candidate"} for value in sorted(excluded_near)])
    write_jsonl(output / "vlm_data/audits/image_split_conflicts.jsonl", [])
    write_jsonl(output / "vlm_data/audits/v2_test_immutable_reuse.jsonl", immutable_rows)
    write_jsonl(output / "vlm_data/audits/v2_milvus_reference_immutable_reuse.jsonl", reference_reuse)
    contract = v2_test_reference_contract(v2_root)
    summary = {
        "schema_version": "agrinet.open-agri-v3-rag-test-informed/v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "test_informed_protocol": True, "rag_evidence": {"path": str(evidence.relative_to(REPO_ROOT)), "class_difficulty_sha256": digest(evidence / "class_difficulty.jsonl"), "aggregation_summary_sha256": digest(evidence / "aggregation_summary.json")},
        "v2_immutable_contract": {**contract, "test_items_verified": len(immutable_rows), "milvus_reference_items_verified": len(reference_reuse), "all_test_image_sha256_match": all(row["v2_file_sha256"] == row["v3_file_sha256"] == row["image_sha256"] for row in immutable_rows), "all_milvus_reference_image_sha256_match": True},
        "counts": {"classes": 217, "known_classes": len(known), "unknown_classes": len(unknown), "dev_classes": len(dev_codes), "dev_images": len(dev_rows), "test_images": len(test_rows), "milvus_reference_images": len(reference_rows), "train_candidate_images": len(candidate_rows), "task_candidates": task_count, "historical_canonical": historical_counts},
        "policy": {"role_selection": "test-informed protocol; raw-base RAG accuracy, Wilson lower bound, candidate capacity, SHA-256 tie-break", "test": "v2 public test and private audit truth copied unchanged; never used for checkpoint, hyperparameter, prompt, or retrieval selection", "train": "only final known train-candidate images are SFT eligible", "dev": "all known classes plus half of trainable unknown classes, five images per class", "reference": "v2 Milvus reference content copied unchanged"},
    }
    write_json(output / "manifests/summary.json", summary)
    write_json(output / "vlm_data/audits/v2_immutable_contract.json", contract)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-root", type=Path, default=V1_ROOT)
    parser.add_argument("--v2-root", type=Path, default=V2_ROOT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--similar-classes", type=Path, default=DEFAULT_SIMILAR)
    parser.add_argument("--phash-cache", type=Path, default=DEFAULT_PHASH_CACHE)
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--phash-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, sort_keys=True))
