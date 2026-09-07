#!/usr/bin/env python3
"""Build the approved, parallel OpenAgri canonical-v1 benchmark line.

The source OpenAgri v2 directory is read-only. This builder reuses its image
bytes and stable evaluation IDs, records both source and canonical codes, and
never maps an old private-truth file during scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v2"
V1_ROOT = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v1"
CATALOG = REPO_ROOT / "outputs/artifacts/datasets/m1-direct-current-hcv-v1/classes.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v2_canonical_v1"
PHASH_CACHE = REPO_ROOT / "outputs/artifacts/datasets/open-agri-v2-formal-phash-cache-v1.jsonl"

sys.path.insert(0, str(REPO_ROOT / "src"))
from agrinet.research.open_agri_v2_canonical.registry import (
    CANONICAL_VERSION,
    MERGED_SOURCE_TO_CANONICAL,
    build_registry_rows,
    canonical_code,
    load_registry,
    registry_digest,
)

_SPEC = importlib.util.spec_from_file_location("open_agri_v1_1_builder", REPO_ROOT / "scripts/data/build_open_agri_v1_1_hcv_base.py")
assert _SPEC and _SPEC.loader
V2_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(V2_HELPERS)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The dataset root itself is a workspace symlink. A relative target based
    # on the checkout path is interpreted from the resolved shared directory
    # and can therefore become dangling. Persist the immutable source target.
    destination.symlink_to(source.resolve())


def render_current_review_document(rows: list[dict[str, Any]], domain: str) -> str:
    """Render only current proposals for human review."""
    selected = [row for row in rows if row["domain"] == domain]
    title = "病害" if domain == "disease" else "虫害"
    lines = [
        f"# OpenAgri canonical-v1 当前规范方案：{title}", "",
        "本文件只呈现待人工确认的 current proposal。历史标签请看同级 ../legacy/ 文件。",
        "请在 review_decisions.jsonl 中记录决定；不要把本文件中的 proposal 当作已批准标签。", "",
        "| Canonical code | Source codes | English | Chinese | Scientific name | Life stage | Decision |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in selected:
        lines.append(f"| {row['canonical_code']} | {', '.join(row['source_codes'])} | {row['canonical_english_name']} | {row['canonical_chinese_name']} | {row['scientific_name'] or '—'} | {row['life_stage']} | [ ] approve / [ ] revise / [ ] reject |")
    lines.extend(["", "## Review checklist", "", "- canonical English and Chinese display names", "- scientific name and lifecycle wording", "- merge decision and lower-code retention", "- rationale against the separate legacy evidence", "", "Reviewer: ____________________    Date: ____________________", ""])
    return "\n".join(lines) + "\n"


def render_legacy_review_document(rows: list[dict[str, Any]], domain: str) -> str:
    """Render only immutable legacy labels and provenance."""
    selected = [row for row in rows if row["domain"] == domain]
    title = "病害" if domain == "disease" else "虫害"
    lines = [
        f"# OpenAgri canonical-v1 Legacy source evidence：{title}", "",
        "本文件只读：标签来自冻结 source catalog，不是当前推荐输出。", "",
        "| Canonical code | Source code | Legacy English | Legacy Chinese | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in selected:
        for source in row["legacy_source_labels"]:
            lines.append(f"| {row['canonical_code']} | {source['source_code']} | {source['legacy_english_name']} | {source['legacy_chinese_name']} | read-only provenance |")
    lines.extend(["", "Use this evidence to review the separate current proposal; do not edit legacy labels.", ""])
    return "\n".join(lines) + "\n"


def review_artifacts(rows: list[dict[str, Any]], output: Path) -> None:
    review_root = output / "taxonomy/review"
    write_text(review_root / "README.md", """# OpenAgri v2 canonical-v1 human taxonomy review

This is a proposal, not an approved release. The documents intentionally
separate the proposed current canonical label from frozen legacy source labels.

Review order:

1. Read current/disease.md and current/pest.md. Confirm class meaning, merge,
   English/Chinese display name, scientific name and life stage.
2. Use legacy/disease.md and legacy/pest.md only as source evidence.
3. Record one decision per canonical class in review_decisions.jsonl.
4. After all 211 rows are approved, an authorized reviewer may update each
   registry row to reviewed/approved and create approval.json from
   approval.template.json. Until then, formal SFT, Milvus and scoring are gated.

Legacy labels are provenance-only. They are retained as typed aliases where
unambiguous, but do not define the current display label.
""")
    write_text(review_root / "current/disease.md", render_current_review_document(rows, "disease"))
    write_text(review_root / "current/pest.md", render_current_review_document(rows, "pest"))
    write_text(review_root / "legacy/disease.md", render_legacy_review_document(rows, "disease"))
    write_text(review_root / "legacy/pest.md", render_legacy_review_document(rows, "pest"))
    decisions = [{
        "canonical_code": row["canonical_code"], "current_proposal": {
            "english": row["canonical_english_name"], "chinese": row["canonical_chinese_name"],
            "scientific_name": row["scientific_name"], "life_stage": row["life_stage"],
        }, "decision": "pending", "reviewer": None, "reviewed_at": None, "notes": None,
    } for row in rows]
    write_jsonl(review_root / "review_decisions.jsonl", decisions)
    write_json(review_root / "approval.template.json", {
        "schema_version": "agrinet.canonical-label-approval/v1", "taxonomy_version": CANONICAL_VERSION,
        "status": "approved", "approved_rows": 211, "required_rows": 211,
        "registry_sha256": "COPY_FROM_APPROVED_REGISTRY", "review_scope": "all canonical classes",
        "reviewer": "REQUIRED", "approved_at": "REQUIRED_ISO8601", "review_record": "taxonomy/review/review_decisions.jsonl",
    })


def separated_taxonomy_artifacts(rows: list[dict[str, Any]], output: Path) -> None:
    """Write current proposals and legacy lineage as deliberately separate views."""
    current_rows = [{
        "schema_version": "agrinet.canonical-current-proposal/v1", "taxonomy_version": CANONICAL_VERSION,
        "domain": row["domain"], **row["current"],
    } for row in rows]
    legacy_rows = [{
        "schema_version": "agrinet.canonical-legacy-lineage/v1", "taxonomy_version": CANONICAL_VERSION,
        "canonical_code": row["canonical_code"], "domain": row["domain"],
        "source_code": source["source_code"], "legacy_english_name": source["legacy_english_name"],
        "legacy_chinese_name": source["legacy_chinese_name"],
    } for row in rows for source in row["legacy_source_labels"]]
    write_jsonl(output / "taxonomy/current/canonical_proposals.jsonl", current_rows)
    write_jsonl(output / "taxonomy/legacy/source_label_lineage.jsonl", legacy_rows)


def source_roles(rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    role_by_source = {str(row["class_code"]): str(row["class_role"]) for row in rows}
    dev_by_source = {str(row["class_code"]): str(row["dev_role"]) for row in rows}
    if len(role_by_source) != 217:
        raise ValueError("source v2 class split must cover 217 source codes")
    return role_by_source, dev_by_source


def canonical_roles(role_by_source: dict[str, str]) -> dict[str, str]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for source_code, role in role_by_source.items():
        grouped[canonical_code(source_code)].add(role)
    roles = {code: ("known" if "known" in values else "unknown") for code, values in grouped.items()}
    for source_code, canonical in {
        "N04031": "N04029", "N04082": "N04080", "N04116": "N04111",
    }.items():
        if roles[canonical] != "known":
            raise ValueError(f"former cross-role merge {source_code}/{canonical} must be Known")
    return roles


def cache_phashes(rows: list[dict[str, Any]], cache: Path, workers: int) -> None:
    """Fill pHashes from the existing immutable cache, only computing misses."""
    helper_rows = [dict(row, sha256=row["image_sha256"], class_code=row["canonical_class_code"]) for row in rows]
    V2_HELPERS.near_duplicate_audit(helper_rows, threshold=6, workers=workers, cache_path=cache)
    for row, helper in zip(rows, helper_rows, strict=True):
        row["phash"] = helper["phash"]


def build_image_rows(
    source_pool: list[dict[str, Any]], canonical_roles_by_code: dict[str, str], workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstitute the source-image split under canonical labels.

    Candidate images are drawn from the v1 training manifest so a source code
    that was formerly v2-Unknown but now belongs to a merged Known class is not
    silently omitted. Dev/test/reference image identities remain from v2.
    """
    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_pool:
        split_rows[str(row["image_split"])].append(dict(row))
    evaluation = split_rows["dev"] + split_rows["test"] + split_rows["milvus_reference"]
    evaluation_hashes = {str(row["image_sha256"]) for row in evaluation}
    candidates: dict[str, dict[str, Any]] = {}
    for row in split_rows["train_candidate"]:
        source_code = str(row["class_code"])
        canonical = canonical_code(source_code)
        if canonical_roles_by_code[canonical] != "known":
            continue
        candidates[str(row["image_sha256"])] = {
            "image_sha256": str(row["image_sha256"]), "source_path": str(row["source_path"]),
            "source_class_code": source_code, "canonical_class_code": canonical,
            "domain": str(row["domain"]), "image_name": Path(str(row["image_path"])).name,
            "image_split": "train_candidate", "class_role": "known",
            "sft_eligible": True, "dev_eligible": False, "milvus_eligible": False,
        }
    # Only these source classes change from source-v2 Unknown to canonical-v1
    # Known. Add their original training candidates without rehashing every
    # v1 class. All other Known candidates already passed v2's pHash audit.
    newly_known_sources = {"N04029", "N04080", "N04111"}
    for source in read_jsonl(V1_ROOT / "manifests/train.jsonl"):
        source_code = str(source["class_code"])
        if source_code not in newly_known_sources:
            continue
        source_path = Path(str(source["source_path"]))
        image_sha256 = digest(source_path)
        if image_sha256 in evaluation_hashes:
            continue
        candidates.setdefault(image_sha256, {
            "image_sha256": image_sha256, "source_path": str(source_path),
            "source_class_code": source_code, "canonical_class_code": canonical_code(source_code),
            "domain": str(source["domain"]), "image_name": str(source["image_name"]),
            "image_split": "train_candidate", "class_role": "known",
            "sft_eligible": True, "dev_eligible": False, "milvus_eligible": False,
        })
    converted_evaluation: list[dict[str, Any]] = []
    for row in evaluation:
        source_code = str(row["class_code"])
        canonical = canonical_code(source_code)
        split = str(row["image_split"])
        candidates_key = "image_sha256"
        converted = {
            "image_sha256": str(row[candidates_key]), "source_path": str(row["source_path"]),
            "source_class_code": source_code, "canonical_class_code": canonical,
            "domain": str(row["domain"]), "image_name": Path(str(row["image_path"])).name,
            "image_split": split, "class_role": canonical_roles_by_code[canonical],
            "sft_eligible": False, "dev_eligible": split == "dev", "milvus_eligible": split == "milvus_reference",
        }
        converted_evaluation.append(converted)
    all_rows = list(candidates.values()) + converted_evaluation
    cache_phashes(all_rows, PHASH_CACHE, workers)
    audit_rows = [dict(row, class_code=row["canonical_class_code"], sha256=row["image_sha256"]) for row in all_rows]
    near_pairs = V2_HELPERS.near_duplicate_audit(audit_rows, threshold=6, workers=workers, cache_path=PHASH_CACHE)
    excluded = {str(row["training_sha256"]) for row in near_pairs}
    materialized = [row for row in all_rows if not (row["image_split"] == "train_candidate" and row["image_sha256"] in excluded)]
    return materialized, near_pairs


def build(output: Path, replace: bool, workers: int) -> dict[str, Any]:
    if output.exists():
        if not replace:
            raise ValueError(f"output exists: {output}; use --replace")
        shutil.rmtree(output)
    catalog_rows = read_jsonl(CATALOG)
    registry_rows = build_registry_rows(catalog_rows)
    source_catalog = {str(row["code"]): row for row in catalog_rows}
    registry_path = output / "taxonomy/canonical_label_registry.jsonl"
    write_jsonl(registry_path, registry_rows)
    registry_sha = registry_digest(registry_rows)
    approval = {
        "schema_version": "agrinet.canonical-label-approval/v1", "taxonomy_version": CANONICAL_VERSION,
        "status": "pending_manual_review", "approved_rows": 0, "required_rows": 211,
        "registry_sha256": registry_sha, "review_scope": "all canonical classes",
        "reviewer": None, "approved_at": None, "review_record": "taxonomy/review/review_decisions.jsonl",
    }
    approval_path = output / "taxonomy/approval.json"
    write_json(approval_path, approval)
    registry = load_registry(registry_path, approval_path)
    review_artifacts(registry_rows, output)
    separated_taxonomy_artifacts(registry_rows, output)
    source_class_rows = read_jsonl(SOURCE_ROOT / "manifests/class_split.jsonl")
    role_by_source, dev_by_source = source_roles(source_class_rows)
    roles = canonical_roles(role_by_source)
    source_pool = read_jsonl(SOURCE_ROOT / "manifests/images.jsonl")
    image_rows, near_pairs = build_image_rows(source_pool, roles, workers)
    split_hashes: dict[str, set[str]] = defaultdict(set)
    try:
        logical_root = output.relative_to(REPO_ROOT)
    except ValueError:
        # datasets/AgriNet-1K is a workspace symlink to shared storage.
        # Keep manifests repo-relative while writing through its resolved path.
        logical_root = Path("datasets/AgriNet-1K") / CANONICAL_VERSION
    for row in image_rows:
        if row["image_sha256"] in split_hashes[row["image_split"]]:
            raise ValueError(f"duplicate image ID within split: {row['image_sha256']}")
        split_hashes[row["image_split"]].add(row["image_sha256"])
        row["image_path"] = str(
            logical_root / "images" / row["image_split"] / row["domain"] /
            row["canonical_class_code"] / row["source_class_code"] / row["image_name"]
        )
        destination = output / "images" / row["image_split"] / row["domain"] / row["canonical_class_code"] / row["source_class_code"] / row["image_name"]
        link(Path(row["source_path"]), destination)
    if any(split_hashes[left] & split_hashes[right] for left in split_hashes for right in split_hashes if left < right):
        raise ValueError("canonical-v1 SHA-256 split isolation failed")
    counts = Counter(row["canonical_class_code"] for row in image_rows if row["image_split"] == "train_candidate")
    classes: list[dict[str, Any]] = []
    for code, reg in sorted(registry.rows_by_code.items()):
        sources = [str(value) for value in reg["source_codes"]]
        role = roles[code]
        class_dev = any(dev_by_source.get(source) in {"known_dev", "unknown_dev"} for source in sources)
        classes.append({
            "canonical_class_code": code, "source_codes": sources, "domain": reg["domain"],
            "class_role": role, "dev_role": "known_dev" if role == "known" and class_dev else ("unknown_dev" if class_dev else ("known" if role == "known" else "unknown_holdout")),
            "sft_eligible": role == "known", "train_candidate_images": counts[code],
            "canonical_english_name": reg["canonical_english_name"], "canonical_chinese_name": reg["canonical_chinese_name"],
            "life_stage": reg["life_stage"],
        })
    if len(classes) != 211 or any(row["class_role"] == "unknown" and row["sft_eligible"] for row in classes):
        raise ValueError("canonical class role construction failed")
    # v2 public records provide stable IDs and image bytes only. Codes enter
    # only canonical-v1 private truth, generated directly from mapped image rows.
    source_public = {str(row["id"]): row for row in read_jsonl(SOURCE_ROOT / "vlm_data/accepted/test_public.jsonl")}
    source_truth = {str(row["id"]): row for row in read_jsonl(SOURCE_ROOT / "vlm_data/accepted/private/test_truth.jsonl")}
    by_hash = {str(row["image_sha256"]): row for row in image_rows}
    public, truth = [], []
    for item_id, item in sorted(source_public.items()):
        source_truth_row = source_truth[item_id]
        image = by_hash[str(item["image_sha256"])]
        code = image["canonical_class_code"]
        public.append({"id": item_id, "images": [image["image_path"]], "image_sha256": image["image_sha256"], "class_role": roles[code], "evaluation_bucket": "known" if roles[code] == "known" else "unknown_holdout", "task": item.get("task", "agricultural_image_classification"), "metadata": {"domain": image["domain"], "source_split": "test", "source_class_code": image["source_class_code"], "canonical_class_code": code}})
        truth.append({"id": item_id, "image_sha256": image["image_sha256"], "source_class_code": image["source_class_code"], "canonical_class_code": code, "domain": image["domain"], "class_role": roles[code], "evaluation_bucket": "known" if roles[code] == "known" else "unknown_holdout"})
    dev_images = [row for row in image_rows if row["image_split"] == "dev"]
    source_dev_public = {str(row["image_sha256"]): row for row in read_jsonl(SOURCE_ROOT / "vlm_data/accepted/dev.jsonl")}
    if set(source_dev_public) != {str(row["image_sha256"]) for row in dev_images}:
        raise ValueError("canonical-v1 dev image inventory must preserve the source-v2 dev IDs")
    dev_public = [{"id": source_dev_public[row["image_sha256"]]["id"], "images": [row["image_path"]], "image_sha256": row["image_sha256"], "class_role": row["class_role"], "evaluation_bucket": "known" if row["class_role"] == "known" else "unknown_dev", "task": "agricultural_image_classification", "metadata": {"domain": row["domain"], "source_split": "dev", "source_class_code": row["source_class_code"], "canonical_class_code": row["canonical_class_code"]}} for row in dev_images]
    dev_truth = [{"id": item["id"], "image_sha256": item["image_sha256"], "source_class_code": item["metadata"]["source_class_code"], "canonical_class_code": item["metadata"]["canonical_class_code"], "domain": item["metadata"]["domain"], "class_role": item["class_role"], "evaluation_bucket": item["evaluation_bucket"]} for item in dev_public]
    canonical_catalog = []
    for row in registry_rows:
        source_codes = [str(value) for value in row["source_codes"]]
        primary = source_catalog[str(row["canonical_code"])]
        canonical_catalog.append({
            **row,
            "source_labels": [{"source_code": code, "english_name": source_catalog[code]["english_name"], "chinese_name": source_catalog[code]["chinese_name"]} for code in source_codes],
            "public_knowledge": str(primary.get("public_knowledge") or ""),
            "source_catalog_sha256": digest(CATALOG),
        })
    write_jsonl(output / "catalog/canonical_catalog.jsonl", canonical_catalog)
    write_jsonl(output / "manifests/class_split.jsonl", classes)
    write_jsonl(output / "manifests/images.jsonl", image_rows)
    write_jsonl(output / "vlm_data/candidates/image_pool.jsonl", image_rows)
    write_jsonl(output / "vlm_data/accepted/dev.jsonl", dev_public)
    write_jsonl(output / "vlm_data/accepted/test_public.jsonl", public)
    write_jsonl(output / "vlm_data/accepted/private/dev_truth.jsonl", dev_truth)
    write_jsonl(output / "vlm_data/accepted/private/test_truth.jsonl", truth)
    write_jsonl(output / "vlm_data/audits/near_duplicate_clusters.jsonl", near_pairs)
    summary = {"schema_version": "agrinet.open-agri-v2-canonical-v1/v1", "taxonomy_version": CANONICAL_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(), "registry_sha256": registry_sha, "source_catalog_sha256": digest(CATALOG), "source_v2_images_sha256": digest(SOURCE_ROOT / "manifests/images.jsonl"), "release_status": "pending_manual_review", "counts": {"source_classes": 217, "canonical_classes": 211, "approved_classes": 0, "known_classes": sum(row["class_role"] == "known" for row in classes), "unknown_classes": sum(row["class_role"] == "unknown" for row in classes), "images": len(image_rows), "test_images": len(public), "dev_images": len(dev_public), "train_candidate_images": sum(row["image_split"] == "train_candidate" for row in image_rows)}, "policy": {"source_v2_immutable": True, "former_cross_role_merges_known": ["N04029/N04031", "N04080/N04082", "N04111/N04116"], "private_truth": "generated directly under canonical codes; old v2 truth is not source-code-mapped while scoring", "formal_release": "blocked until all review decisions and approval.json are signed"}}
    write_json(output / "manifests/summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--phash-workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    print(json.dumps(build(args.output_root, args.replace, args.phash_workers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
