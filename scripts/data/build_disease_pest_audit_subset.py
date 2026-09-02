#!/usr/bin/env python3
"""Build the non-overlapping disease/pest audit subdataset.

The source-of-truth image audit is the ``categories`` section of
img_1k_select_feedback_record.json.  This tool deliberately does not modify
the source ``all/`` tree, the historical open_domain split, or Milvus.

The resulting ``test`` is a frozen evaluation set.  Its source audit state is
named ``validation``, but it must not be used for model/RAG selection or
prompt tuning.  When a downstream experiment selects only some classes for
model training, split this fixed test manifest by class into ``train_seen``
and ``train_unseen_kb_covered`` reports.  The latter means unseen by model
training, not unknown to the Milvus knowledge base.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def relative_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(os.path.relpath(source, target.parent))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_rows(dataset_root: Path) -> tuple[list[dict], dict[str, set[str]]]:
    wiki_root = dataset_root / "wiki_img/agri_disease_pest_wiki"
    feedback = json.loads((wiki_root / "img_1k_select_feedback_record.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    excluded_original: dict[str, set[str]] = {status: set() for status in ("keep", "validation", "delete")}

    for category in feedback["categories"]:
        code = category["category_id"]
        domain = "disease" if code.startswith("N04") else "pest"
        for source_name in ("img_1k_select", "other_source"):
            for status, names in category[source_name].items():
                for image_name in names:
                    if source_name == "img_1k_select":
                        image_path = dataset_root / "all" / code / image_name
                    else:
                        matches = list((wiki_root / "img" / domain).glob(f"{code}*/{image_name}"))
                        image_path = matches[0] if len(matches) == 1 else None
                    exists = image_path is not None and image_path.is_file()
                    row = {
                        "class_code": code,
                        "domain": domain,
                        "audit_status": status,
                        "audit_source": source_name,
                        "image_name": image_name,
                        "source_path": str(image_path) if image_path else None,
                        "source_exists": exists,
                    }
                    if exists:
                        row["sha256"] = sha256(image_path)
                    rows.append(row)
                    if source_name == "img_1k_select":
                        excluded_original[status].add(str(image_path.resolve()))
    return rows, excluded_original


def link_manifest_rows(rows: list[dict], split_root: Path, split_name: str) -> None:
    for row in rows:
        source = Path(row["source_path"])
        target = split_root / row["domain"] / split_name / row["class_code"] / row["image_name"]
        relative_link(source, target)
        row["subset_path"] = str(target)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    all_root = dataset_root / "all"
    if not all_root.is_dir():
        raise SystemExit(f"Missing source directory: {all_root}")
    if output_root.exists():
        if not args.replace:
            raise SystemExit(f"Output already exists: {output_root}; use --replace after review")
        shutil.rmtree(output_root)

    audit, original_excluded = audit_rows(dataset_root)
    invalid = [row for row in audit if row["audit_status"] != "delete" and not row["source_exists"]]
    if invalid:
        raise SystemExit(f"{len(invalid)} non-delete audit rows have no local image; first={invalid[0]}")

    keep_rows = [row for row in audit if row["audit_status"] == "keep"]
    # Keep has precedence over validation.  Four external validation records
    # duplicate keep images byte-for-byte; retaining them as test data would
    # leak Milvus reference content into the test set.  Deduplicate remaining
    # validation rows by hash as well, preserving the first stable record.
    keep_hashes = {row["sha256"] for row in keep_rows}
    test_rows: list[dict] = []
    test_hashes: set[str] = set()
    test_conflicts: list[dict] = []
    for row in (row for row in audit if row["audit_status"] == "validation"):
        conflict_reason = None
        if row["sha256"] in keep_hashes:
            conflict_reason = "duplicate_of_milvus_reference"
        elif row["sha256"] in test_hashes:
            conflict_reason = "duplicate_validation_content"
        if conflict_reason:
            test_conflicts.append({**row, "exclusion_reason": conflict_reason})
            continue
        test_hashes.add(row["sha256"])
        test_rows.append(row)
    delete_rows = [row for row in audit if row["audit_status"] == "delete"]
    link_manifest_rows(keep_rows, output_root, "milvus_reference")
    link_manifest_rows(test_rows, output_root, "test")

    # Training comprises only original disease/pest images not selected in any
    # audit state.  No keep, test, or delete image can enter training.
    audit_original_paths = set().union(*original_excluded.values())
    train_rows: list[dict] = []
    for class_dir in sorted(path for path in all_root.iterdir() if path.is_dir() and path.name.startswith(("N04", "N05"))):
        domain = "disease" if class_dir.name.startswith("N04") else "pest"
        for image in sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
            if str(image.resolve()) in audit_original_paths:
                continue
            row = {
                "class_code": class_dir.name,
                "domain": domain,
                "source_path": str(image),
                "image_name": image.name,
            }
            target = output_root / domain / "train" / class_dir.name / image.name
            relative_link(image, target)
            row["subset_path"] = str(target)
            train_rows.append(row)

    # Hash validation is necessary for the small audited sets.  Training is
    # path-excluded from every original audit row above, so hashing all roughly
    # 194k remaining source images would add a large, unnecessary I/O pass.
    by_hash: dict[str, set[str]] = defaultdict(set)
    for name, rows in (("milvus_reference", keep_rows), ("test", test_rows)):
        for row in rows:
            by_hash[row["sha256"]].add(name)
    overlaps = {key: sorted(value) for key, value in by_hash.items() if len(value) > 1}
    if overlaps:
        raise SystemExit(f"Cross-split SHA-256 overlap detected: {next(iter(overlaps.items()))}")

    manifests = output_root / "manifests"
    write_jsonl(manifests / "milvus_reference.jsonl", keep_rows)
    write_jsonl(manifests / "test.jsonl", test_rows)
    write_jsonl(manifests / "test_content_conflicts.jsonl", test_conflicts)
    write_jsonl(manifests / "train.jsonl", train_rows)
    write_jsonl(manifests / "delete.jsonl", delete_rows)
    write_jsonl(manifests / "audit_all.jsonl", audit)

    rows_by_purpose = {"milvus_reference": keep_rows, "test": test_rows, "train": train_rows, "delete": delete_rows}
    summary = {
        "schema_version": "agrinet.disease-pest-audit-subset/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_root": str(dataset_root),
        "audit_record": str(dataset_root / "wiki_img/agri_disease_pest_wiki/img_1k_select_feedback_record.json"),
        "policy": {
            "milvus_reference": "audit keep only; excluded from train and test",
            "test": "frozen final evaluation set; sourced from audit validation, excluded from train and Milvus reference, and not for tuning",
            "train": "original disease/pest all/ images with every audit-selected image excluded",
            "delete": "excluded from all subset views; source all/ is retained unchanged for historical reproducibility",
        },
        "rag_evaluation_protocol": {
            "knowledge_base_scope": "milvus_reference covers all 217 disease/pest classes and is external retrieval knowledge",
            "training_scope": "a downstream experiment may select only a subset of train classes for model training",
            "test_reporting": {
                "train_seen": "test images whose class is included in the experiment training-class list",
                "train_unseen_kb_covered": "test images whose class is absent from the experiment training-class list but remains covered by Milvus",
            },
            "terminology": "train-unseen is relative to model training only; it is not an open-set or knowledge-base-unknown label",
            "test_freeze": "do not use the fixed test set for checkpoint selection, hyperparameter selection, retrieval tuning, or prompt tuning; use a separate development validation set",
        },
        "counts": {
            purpose: {
                "images": len(rows),
                "classes": len({row["class_code"] for row in rows}),
                "disease_images": sum(row["domain"] == "disease" for row in rows),
                "pest_images": sum(row["domain"] == "pest" for row in rows),
            }
            for purpose, rows in rows_by_purpose.items()
        },
        "cross_split_sha256_overlaps": 0,
        "test_rows_excluded_for_content_overlap": len(test_conflicts),
        "source_all_mutated": False,
    }
    (manifests / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (manifests / "class_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_code", "domain", "milvus_reference", "test", "train", "delete"])
        writer.writeheader()
        for code in sorted({row["class_code"] for row in audit}):
            domain = "disease" if code.startswith("N04") else "pest"
            writer.writerow({
                "class_code": code, "domain": domain,
                **{purpose: sum(row["class_code"] == code for row in rows) for purpose, rows in rows_by_purpose.items()},
            })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
