#!/usr/bin/env python3
"""Build real-data inputs and a fail-closed capacity audit for M1 Direct."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agrinet.research.m1.collection import build_plan, image_sha256
from agrinet.research.shared.catalog import exposed_images, read_jsonl, write_jsonl
from agrinet.research.hcv.retrieval_preflight import CURRENT_SFT_SOURCES, FORMAL_618, _row_image

SUPPLEMENTAL = ROOT / "outputs/artifacts/datasets/m1-direct-current-hcv-v1/n05053_candidates/approved_manifest.jsonl"


def canonical_classes() -> list[dict[str, str]]:
    payload = json.loads((ROOT / "datasets/AgriNet-1K/wiki/base.json").read_text(encoding="utf-8"))
    image_root = ROOT / "datasets/AgriNet-1K/all"
    training_codes = {
        path.name for path in image_root.iterdir()
        if path.is_dir() and path.name.startswith(("N04", "N05"))
    }
    by_code: dict[str, dict[str, str]] = {}
    for entry in (payload.get("description") or {}).values():
        code = str(entry.get("code") or "")
        if code not in training_codes:
            continue
        descriptions = entry.get("description") if isinstance(entry.get("description"), dict) else {}
        knowledge = "\n\n".join(str(value).strip() for value in descriptions.values() if str(value).strip())
        if code not in by_code:
            by_code[code] = {
                "code": code,
                "task_domain": "disease" if code.startswith("N04") else "pest",
                "english_name": str(entry.get("english_name") or "").strip(),
                "chinese_name": str(entry.get("chinese_name") or "").strip(),
                "public_knowledge": knowledge,
            }
        elif knowledge and knowledge not in by_code[code]["public_knowledge"]:
            by_code[code]["public_knowledge"] += "\n\n" + knowledge
    missing = training_codes - set(by_code)
    if missing:
        raise ValueError(f"training classes missing from public wiki: {sorted(missing)}")
    return [by_code[code] for code in sorted(by_code)]


def similar_rows(
    classes: list[dict[str, str]], *, audit: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Choose three same-domain joint neighbours with unambiguous zh views.

    The source is a joint image/text top-12 ordering.  We preserve that ordering
    but skip a candidate only when its Chinese display name duplicates the anchor
    or an already selected candidate.  This keeps Chinese Open answers and
    Chinese Option text uniquely answerable without changing a class label or
    relaxing the three-hard-negative requirement.
    """
    path = ROOT / "outputs/milvus/wiki_similar_classes_siglip2_report.top12.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    domains = {row["code"]: row["task_domain"] for row in classes}
    by_code: dict[str, dict[str, Any]] = {}
    for row in report.get("classes") or []:
        code = str(row.get("code") or "")
        if code not in domains:
            continue
        candidates = []
        skipped_duplicate_names: list[dict[str, str]] = []
        anchor_name = str(next(item["chinese_name"] for item in classes if item["code"] == code))
        used_zh_names = {anchor_name}
        for item in row.get("similar") or []:
            candidate = str(item.get("code") or "")
            if candidate in domains and domains[candidate] == domains[code] and candidate != code:
                candidate_name = str(next(value["chinese_name"] for value in classes if value["code"] == candidate))
                if candidate in candidates:
                    continue
                if candidate_name in used_zh_names:
                    skipped_duplicate_names.append({"code": candidate, "chinese_name": candidate_name})
                    continue
                if candidate not in candidates:
                    candidates.append(candidate)
                    used_zh_names.add(candidate_name)
                if len(candidates) == 3:
                    break
        candidate_row = {
            "class_code": code,
            "hard_negative_codes": candidates[:3],
            "source": "wiki_siglip2_joint_text_image_top12_zh_unique",
        }
        if audit is not None and skipped_duplicate_names:
            audit.append({
                "class_code": code,
                "anchor_chinese_name": anchor_name,
                "original_top3": [str(item.get("code") or "") for item in (row.get("similar") or [])[:3]],
                "skipped_duplicate_chinese_names": skipped_duplicate_names,
                "selected_hard_negative_codes": candidates[:3],
                "source": candidate_row["source"],
            })
        current = by_code.get(code)
        if current is None or len(candidate_row["hard_negative_codes"]) > len(current["hard_negative_codes"]):
            by_code[code] = candidate_row
    return [by_code[code] for code in sorted(by_code)]


def hashes_from_rows(path: Path) -> set[str]:
    hashes: set[str] = set()
    if not path.is_file():
        return hashes
    for row in read_jsonl(path):
        image = _row_image(row)
        if not image:
            continue
        candidate = Path(image)
        candidate = candidate if candidate.is_absolute() else ROOT / candidate
        try:
            hashes.add(image_sha256(candidate))
        except OSError:
            continue
    return hashes


def isolation_sets() -> dict[str, set[str]]:
    formal = hashes_from_rows(FORMAL_618)
    historical: set[str] = set()
    current_rag: set[str] = set()
    for path in CURRENT_SFT_SOURCES:
        values = hashes_from_rows(path)
        historical.update(values)
        if "rag" in str(path).lower() or "hcv" in str(path).lower():
            current_rag.update(values)
    contacted: set[str] = set()
    for image in exposed_images():
        candidate = Path(image)
        candidate = candidate if candidate.is_absolute() else ROOT / candidate
        try:
            contacted.add(image_sha256(candidate))
        except OSError:
            continue
    return {
        "formal_evaluation": formal,
        "historical_training": historical,
        "current_rag": current_rag,
        "contacted_or_retired": contacted,
    }


def image_pool(classes: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for item in classes:
        directory = ROOT / "datasets/AgriNet-1K/all" / item["code"]
        for path in sorted(directory.glob("*")):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            rows.append({
                "class_code": item["code"],
                "query_image": str(path.relative_to(ROOT)),
                "image_sha256": image_sha256(path),
            })
    if SUPPLEMENTAL.is_file():
        seen_occurrences: set[str] = set()
        for row in read_jsonl(SUPPLEMENTAL):
            path = ROOT / str(row.get("query_image") or "")
            occurrence = str(row.get("occurrence_id") or "")
            if row.get("class_code") != "N05053" or not occurrence or occurrence in seen_occurrences:
                raise ValueError("invalid or duplicate N05053 supplemental occurrence")
            if not str(row.get("license") or "").startswith("cc-") or not row.get("visual_review"):
                raise ValueError(f"unlicensed or unreviewed N05053 supplemental image: {occurrence}")
            digest = image_sha256(path)
            if digest != row.get("image_sha256"):
                raise ValueError(f"N05053 supplemental hash mismatch: {occurrence}")
            seen_occurrences.add(occurrence)
            rows.append({"class_code": "N05053", "query_image": str(path.relative_to(ROOT)), "image_sha256": digest})
    return rows


def build(output: Path) -> dict[str, Any]:
    classes = canonical_classes()
    similar_audit: list[dict[str, Any]] = []
    similar = similar_rows(classes, audit=similar_audit)
    images = image_pool(classes)
    isolated = isolation_sets()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "classes.jsonl", classes)
    write_jsonl(output / "similar_classes.jsonl", similar)
    write_jsonl(output / "reports" / "hard_negative_zh_uniqueness_repairs.jsonl", similar_audit)
    write_jsonl(output / "image_pool.jsonl", images)
    isolation_dir = output / "isolation"
    for name, values in isolated.items():
        write_jsonl(isolation_dir / f"{name}.jsonl", [{"image_sha256": value} for value in sorted(values)])
    excluded = set().union(*isolated.values())
    available = Counter(row["class_code"] for row in images if row["image_sha256"] not in excluded)
    shortages = {
        row["code"]: max(0, 5 - available[row["code"]])
        for row in classes if available[row["code"]] < 5
    }
    similar_shortages = {
        row["class_code"]: 3 - len(row["hard_negative_codes"])
        for row in similar if len(row["hard_negative_codes"]) < 3
    }
    report = {
        "schema_version": "agrinet.m1-direct-preflight/v1",
        "classes": len(classes),
        "source_images": len(images),
        "source_unique_hashes": len({row["image_sha256"] for row in images}),
        "isolation_counts": {key: len(value) for key, value in isolated.items()},
        "isolation_union": len(excluded),
        "available_images": sum(available.values()),
        "min_available_per_class": min((available[row["code"]] for row in classes), default=0),
        "shortages": shortages,
        "similar_class_shortages": similar_shortages,
        "similar_classes_complete": len(similar) == len(classes) and all(
            len(row["hard_negative_codes"]) == 3 for row in similar
        ),
        "hard_negative_zh_uniqueness_repairs": len(similar_audit),
        "plan_ready": len(classes) == 217 and not shortages and not similar_shortages,
        "teacher_authorized": False,
    }
    if report["plan_ready"]:
        _, _, plan_report = build_plan(
            classes, images,
            {row["class_code"]: row["hard_negative_codes"] for row in similar},
            isolated,
        )
        report["planned_rows"] = plan_report["teacher_rows"]
    report_path = output / "preflight_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/artifacts/datasets/m1-direct-current-hcv-v1",
    )
    args = parser.parse_args()
    report = build(args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["plan_ready"] and report["similar_classes_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
