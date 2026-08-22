#!/usr/bin/env python3
"""Plan fresh, hash-isolated Direct and Blind RAG targets after v2 audits."""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.data.rebuild_sft import CELLS, canonical_json_hash, image_digest, load_hermes_1to1_specification
from agrinet.data.sft_recovery import read_jsonl, write_jsonl

IMAGE_ROOT = Path("datasets/AgriNet-1K/all")
SEED = 20260817


def catalog(path: Path) -> dict[str, dict[str, str]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("classes") if isinstance(payload, dict) else None
        if not isinstance(rows, list): raise ValueError(f"catalog JSON lacks classes: {path}")
        return {str(row["code"]): {"code": str(row["code"]), "english_name": str(row.get("english_name") or ""), "chinese_name": str(row.get("chinese_name") or ""), "task_domain": "disease" if str(row["code"]).startswith("N04") else "pest" if str(row["code"]).startswith("N05") else ""} for row in rows if row.get("code")}
    return {str(row["code"]): {key: str(row.get(key) or "") for key in ("code", "english_name", "chinese_name", "task_domain")} for row in read_jsonl(path)}


def option_labels(classes: dict[str, dict[str, str]], code: str, domain: str, correct_option: str) -> list[dict[str, str]]:
    alternatives = sorted((entry for candidate, entry in classes.items() if entry["task_domain"] == domain and candidate != code), key=lambda entry: entry["code"])[:3]
    alternatives.insert(ord(correct_option) - ord("A"), classes[code])
    return [{"code": entry["code"], "name": entry["english_name"], "chinese_name": entry["chinese_name"], "task_domain": entry["task_domain"]} for entry in alternatives]


def image_index(classes: dict[str, dict[str, str]], *, root: Path, forbidden: set[str]) -> dict[str, list[tuple[str, str]]]:
    """Hash every candidate image once; selection must never rescan a class."""
    index: dict[str, list[tuple[str, str]]] = {}
    for code in sorted(classes):
        images = []
        # We need no more than two images/class in any cell.  Hashing the full
        # corpus is unnecessary and makes planning non-resumable on a large
        # local image store; a stable first-24 window leaves ample reserve.
        paths = sorted(itertools.islice((root / IMAGE_ROOT / code).iterdir(), 24)) if (root / IMAGE_ROOT / code).is_dir() else []
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}: continue
            image = str(path.relative_to(root)); digest = image_digest(image, root)
            if digest not in forbidden: images.append((image, digest))
        index[code] = images
    return index


def choose(cell: tuple[str, str, str], *, count: int, classes: dict[str, dict[str, str]], image_candidates: dict[str, list[tuple[str, str]]], used: set[str]) -> list[tuple[str, str, str]]:
    _, _, domain = cell
    codes = [code for code, entry in classes.items() if entry["task_domain"] == domain and image_candidates.get(code)]
    random.Random(f"{SEED}:{'/'.join(cell)}").shuffle(codes)
    selected: list[tuple[str, str, str]] = []; class_count: Counter[str] = Counter()
    # One per class first, then a second round; with 70 rows this guarantees
    # the 35-class floor and two-row cap if the image corpus is sufficient.
    for round_index in range(2):
        for code in codes:
            if len(selected) >= count: break
            if class_count[code] > round_index: continue
            candidate = next(((image, digest) for image, digest in image_candidates[code] if digest not in used), None)
            if candidate is None: continue
            image, digest = candidate
            selected.append((code, image, digest)); used.add(digest); class_count[code] += 1
        if len(selected) >= count: break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--forbidden-hashes", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    spec, entries = load_hermes_1to1_specification(), catalog(args.classes)
    forbidden = set(json.loads(args.forbidden_hashes.read_text(encoding="utf-8"))); used: set[str] = set()
    image_candidates = image_index(entries, root=args.repo_root, forbidden=forbidden)
    direct, rag, shortages = [], [], {}
    # Direct uses one image for an English/Chinese pair; each question type and
    # domain gets its own 70 images, so no image has more than the required pair.
    for question_type in ("open", "option"):
        for domain in ("disease", "pest"):
            chosen = choose((question_type, "en", domain), count=spec["direct_per_cell"], classes=entries, image_candidates=image_candidates, used=used)
            shortages[f"direct/{question_type}/{domain}"] = spec["direct_per_cell"] - len(chosen)
            for target_index, (code, image, digest) in enumerate(chosen, 1):
                for language in ("en", "zh"):
                    target = {"target_id": f"hermes-v2-direct-{question_type}-{language}-{domain}-{target_index:03d}", "query_image": image, "image_sha256": digest, "canonical_class": code, "canonical_name": entries[code]["chinese_name"] if language == "zh" else entries[code]["english_name"], "task_domain": domain, "question_type": question_type, "language": language, "paired_image_id": digest, "generation_route": "direct_visual_comparison", "label_visible_to_teacher": True, "uncertainty": spec["uncertainty"], "teacher_requirements": {"visible_candidates_min": spec["direct_min_candidates"], "neighbor_exclusions_min": spec["direct_min_negative_exclusions"], "preserve_long_comparison": True}}
                    if question_type == "option": target["correct_option"] = "ABCD"[(target_index - 1) % 4]; target["candidate_labels"] = option_labels(entries, code, domain, target["correct_option"])
                    direct.append(target)
    for cell in CELLS:
        question_type, language, domain = cell
        chosen = choose(cell, count=spec["rag_per_cell"], classes=entries, image_candidates=image_candidates, used=used)
        shortages[f"rag/{'/'.join(cell)}"] = spec["rag_per_cell"] - len(chosen)
        for target_index, (code, image, digest) in enumerate(chosen, 1):
            target = {"target_id": f"hermes-v2-rag-{question_type}-{language}-{domain}-{target_index:03d}", "query_image": image, "image_sha256": digest, "canonical_class": code, "canonical_name": entries[code]["chinese_name"] if language == "zh" else entries[code]["english_name"], "task_domain": domain, "question_type": question_type, "language": language, "generation_route": "blind_evidence", "label_visible_to_teacher": False, "uncertainty": spec["uncertainty"], "teacher_requirements": {"pre_tool_visual_candidates_min": spec["rag_min_candidates"], "evidence_exclusions_min": spec["rag_min_evidence_exclusions"], "public_tool_evidence_only": True}}
            if question_type == "option": target["correct_option"] = "ABCD"[(target_index - 1) % 4]; target["candidate_labels"] = option_labels(entries, code, domain, target["correct_option"])
            rag.append(target)
    report = {"schema_version": "agrinet.hermes-1to1-shortage-plan/v1", "seed": SEED, "direct_rows": len(direct), "rag_rows": len(rag), "unique_images": len(used), "shortages": shortages, "direct_sha256": canonical_json_hash(direct), "rag_sha256": canonical_json_hash(rag), "training_authorized": False, "next_gate": "teacher collection only after review of this immutable plan"}
    args.destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.destination / "direct_shortage_targets.jsonl", direct)
    write_jsonl(args.destination / "blind_rag_shortage_targets.jsonl", rag)
    (args.destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not any(shortages.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
