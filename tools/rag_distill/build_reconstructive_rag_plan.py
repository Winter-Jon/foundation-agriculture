#!/usr/bin/env python3
"""Plan image-isolated Blind RAG collection; never invoke a teacher."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agrinet.data.rebuild_sft import CELLS, STAGE_SPECS, image_digest, load_collection_specification, minimum_classes, next_trajectory
from agrinet.data.sft_recovery import read_jsonl, write_jsonl
from tools.rag_distill.catalog_and_isolation import ROOT, candidate_labels, exposed_images
from tools.rag_distill.build_reconstructive_direct_candidates import isolation_hashes

CANDIDATES = ROOT / "outputs/vlm_data/disease_pest_large/contrast_samples_vit_base.jsonl"
CLASSES = ROOT / "outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl"
MILVUS_CATALOG = ROOT / "outputs/milvus/wiki_similar_classes_siglip2_report.json"
IMAGE_ROOT = ROOT / "datasets/AgriNet-1K/all"
SEED = 20260816
RECONSTRUCTIVE_ROOT = ROOT / "outputs/experiments/reconstructive_direct_blind_rag_v1"


def catalog(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        for label in row.get("candidate_labels") or []:
            code = str(label.get("code") or "")
            if code:
                result.setdefault(code, {"code": code, "english_name": str(label.get("name") or ""), "chinese_name": str(label.get("chinese_name") or ""), "task_domain": str(label.get("task_domain") or row.get("task_domain") or "")})
    return result


def raw_catalog() -> dict[str, dict[str, str]]:
    if MILVUS_CATALOG.is_file():
        report = json.loads(MILVUS_CATALOG.read_text(encoding="utf-8"))
        result = {}
        for row in report.get("classes") or []:
            code = str(row.get("code") or "")
            if code and row.get("english_name") and row.get("chinese_name"):
                result[code] = {"code": code, "english_name": str(row["english_name"]), "chinese_name": str(row["chinese_name"]), "task_domain": "disease" if code.startswith("N04") else "pest" if code.startswith("N05") else ""}
        if result:
            return result
    return {str(row["code"]): {"code": str(row["code"]), "english_name": str(row["english_name"]), "chinese_name": str(row["chinese_name"]), "task_domain": str(row["task_domain"])} for row in read_jsonl(CLASSES)}


def fresh_images_for_class(code: str, forbidden: set[str], used: set[str], root: Path, limit: int) -> list[tuple[str, str]]:
    directory = IMAGE_ROOT / code
    selected: list[tuple[str, str]] = []
    for path in sorted(directory.glob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = str(path.relative_to(root))
        if image in used:
            continue
        digest = image_digest(image, root)
        if digest in forbidden:
            continue
        selected.append((image, digest))
        if len(selected) == limit:
            break
    return selected


def contacted_reconstructive_hashes(root: Path) -> set[str]:
    """Return historical hashes for lineage only, never as a blanket exclusion."""
    hashes: set[str] = set()
    if not RECONSTRUCTIVE_ROOT.exists():
        return hashes
    for path in RECONSTRUCTIVE_ROOT.rglob("*.jsonl"):
        try:
            rows = read_jsonl(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for row in rows:
            sample = row.get("sample") or row.get("trace", {}).get("sample") or row
            digest = str(sample.get("image_sha256") or row.get("image_sha256") or "")
            if digest:
                hashes.add(digest)
            image = str(sample.get("query_image") or "")
            if image:
                try:
                    hashes.add(image_digest(image, root))
                except Exception:
                    continue
    for path in RECONSTRUCTIVE_ROOT.rglob("preflight_report.json"):
        try:
            digest = str(json.loads(path.read_text(encoding="utf-8")).get("query_image_sha256") or "")
        except (OSError, json.JSONDecodeError):
            continue
        if digest:
            hashes.add(digest)
    return hashes


def build(stage: str, direct_file: Path, destination: Path, root: Path = ROOT, targets_per_cell: int | None = None, collection_spec_file: Path | None = None) -> dict[str, Any]:
    collection_specification = load_collection_specification(collection_spec_file) if collection_spec_file else STAGE_SPECS
    spec = collection_specification[stage]
    target_count = targets_per_cell or spec["rag_per_cell"]
    if target_count < spec["rag_per_cell"] or target_count > spec["blind_attempt_cap"]:
        raise ValueError("targets_per_cell must be between final quota and Blind attempt cap")
    source = read_jsonl(CANDIDATES)
    classes = raw_catalog()
    # Historical contact alone does not disqualify an image: retries are new
    # independent teacher trajectories. Only final selection and fixed held-out
    # sets are isolating. This static planner has no final candidate freeze yet.
    forbidden = isolation_hashes(root)
    for row in read_jsonl(direct_file):
        forbidden.add(str((row.get("metadata") or {}).get("image_sha256") or image_digest(str((row.get("images") or [""])[0]), root)))
    excluded = Counter()
    selected: list[dict[str, Any]] = []
    shortages: dict[str, int] = {}
    used_images: set[str] = set()
    for question_type, language, task_domain in CELLS:
        codes = sorted(code for code, entry in classes.items() if entry["task_domain"] == task_domain and (IMAGE_ROOT / code).is_dir())
        random.Random(f"{SEED}:classes:{stage}:{question_type}:{language}:{task_domain}").shuffle(codes)
        chosen: list[dict[str, Any]] = []
        chosen_codes: list[str] = []
        # First secure the declared class floor with one isolated image per
        # class.  A sparse class may not prevent a later valid class from
        # filling this quota.
        for code in codes:
            images = fresh_images_for_class(code, forbidden, used_images, root, 1)
            if not images:
                excluded["insufficient_fresh_images"] += 1
                continue
            image, digest = images[0]
            used_images.add(image); chosen_codes.append(code)
            chosen.append({"sample_id": f"raw-{code}-{Path(image).stem}", "query_image": image, "image_sha256": digest, "final_label": code, "final_label_zh": classes[code]["chinese_name"]})
            if len(chosen_codes) == minimum_classes(stage, task_domain, collection_specification):
                break
        # Reserve attempts are not training rows. Round-robin the full public
        # catalog so later strict acceptance selection can still meet its own
        # class/cap constraints without re-contacting an image.
        cycle_codes = [*chosen_codes, *[code for code in codes if code not in chosen_codes]]
        while len(chosen) < target_count:
            progressed = False
            for code in cycle_codes:
                if len(chosen) >= target_count:
                    break
                images = fresh_images_for_class(code, forbidden, used_images, root, 1)
                if not images:
                    continue
                image, digest = images[0]
                used_images.add(image); progressed = True
                chosen.append({"sample_id": f"raw-{code}-{Path(image).stem}", "query_image": image, "image_sha256": digest, "final_label": code, "final_label_zh": classes[code]["chinese_name"]})
            if not progressed:
                excluded["insufficient_fresh_images"] += 1
                break
        key = f"{question_type}/{language}/{task_domain}"
        shortages[key] = max(0, target_count - len(chosen))
        if len({str(row["final_label"]) for row in chosen}) < minimum_classes(stage, task_domain, collection_specification):
            shortages[key] = max(shortages[key], minimum_classes(stage, task_domain, collection_specification) - len({str(row["final_label"]) for row in chosen}))
        for index, row in enumerate(chosen, 1):
            code = str(row["final_label"])
            lineage = next_trajectory(row["image_sha256"], [])
            assert lineage is not None
            target = {"target_id": f"rebuild-{stage}-{question_type}-{language}-{task_domain}-{index:03d}", "source_sample_id": row["sample_id"], "query_image": row["query_image"], "image_sha256": row["image_sha256"], "task_domain": task_domain, "language": language, "question_type": question_type, "trajectory_mode": "standard", "generation_route": "blind_evidence", "label_visible_to_teacher": False, "canonical_class": code, "class_name": classes[code]["english_name"], "class_name_zh": classes[code]["chinese_name"], "final_label": code, "final_label_zh": row.get("final_label_zh"), "approval_only": True, "approval_scope": "reconstructive_blind_calibration", "candidate_index": 1, "attempt_budget": {"blind": spec["blind_attempt_cap"], "oracle": spec["oracle_attempt_cap"]}, "train_eligible": False, **lineage}
            if question_type == "option":
                letter = "ABCD"[(index - 1) % 4]
                target.update({"correct_option": letter, "candidate_labels": candidate_labels(classes, code, task_domain, letter)})
            selected.append(target)
    coverage = Counter((row["question_type"], row["language"], row["task_domain"]) for row in selected)
    class_coverage = {"/".join(cell): len({row["canonical_class"] for row in selected if (row["question_type"], row["language"], row["task_domain"]) == cell}) for cell in CELLS}
    report = {"schema_version": "agrinet.reconstructive-rag-plan/v2", "stage": stage, "teacher_model": "gpt-5.6-terra", "seed": SEED, "candidate_source": "fresh_images_from_canonical_class_directories", "canonical_catalog": str(MILVUS_CATALOG.relative_to(ROOT)) if MILVUS_CATALOG.is_file() else str(CLASSES.relative_to(ROOT)), "targets": len(selected), "unique_images": len(used_images), "shortages": shortages, "coverage": {"/".join(cell): coverage[cell] for cell in CELLS}, "class_coverage": class_coverage, "exclusions": dict(excluded), "forbidden_image_hashes": len(forbidden), "contacted_reconstructive_hashes": len(contacted_reconstructive_hashes(root)), "calibration": {"attempts_per_cell": 12, "minimum_accepted": 3, "minimum_acceptance_rate": 0.25}, "training_authorized": False, "next_gate": "per-cell Blind calibration only; unknown delivery may receive one independent resend"}
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "blind_targets.jsonl", selected)
    calibration_targets = [
        row for cell in CELLS for row in selected
        if (row["question_type"], row["language"], row["task_domain"]) == cell
    ]
    calibration_targets = [
        row for cell in CELLS for row in calibration_targets
        if (row["question_type"], row["language"], row["task_domain"]) == cell
    ]
    # Stable per-cell first 12; no calibration result can reshuffle a later
    # target list or replay a delivery of unknown status.
    calibration_targets = [row for cell in CELLS for row in [item for item in selected if (item["question_type"], item["language"], item["task_domain"]) == cell][:12]]
    write_jsonl(destination / "calibration_targets.jsonl", calibration_targets)
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGE_SPECS), required=True)
    parser.add_argument("--direct-file", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--targets-per-cell", type=int)
    parser.add_argument("--collection-spec", type=Path, help="Current collection specification; sampling contract remains code-stable.")
    args = parser.parse_args()
    report = build(args.stage, args.direct_file, args.destination, targets_per_cell=args.targets_per_cell, collection_spec_file=args.collection_spec)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not any(report["shortages"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
