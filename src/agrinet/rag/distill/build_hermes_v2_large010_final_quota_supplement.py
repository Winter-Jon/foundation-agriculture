#!/usr/bin/env python3
"""Produce a fresh final quota repair from the strict v3 preflight.

Unlike large-009, this selects classes already represented once in the v3 view,
so the final selector can add a second image without reducing class coverage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agrinet.data.rebuild_sft import canonical_json_hash, image_digest
from agrinet.data.sft_recovery import write_jsonl
from agrinet.rag.distill.build_hermes_1to1_collection_shard import contacted_hashes
from agrinet.rag.distill.build_hermes_v2_large009_targeted_supplement import catalog
from agrinet.rag.distill.build_hermes_1to1_large_plan import labels

ROOT = Path(__file__).resolve().parents[4]
COLLECTION = ROOT / "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection"
ISOLATION = ROOT / "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/isolation/forbidden_image_sha256.json"
DEST = ROOT / "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-20260819-large-010/supplement"
IMAGE_ROOT = ROOT / "datasets/AgriNet-1K/all"

# These are v3 singleton classes in exactly the cells that remain short.
# Each requested image is new, globally isolated and independent.
DIRECT = {
    ("option", "disease"): ["N04001", "N04042", "N04069", "N04075"],
    ("option", "pest"): ["N05011", "N05018", "N05044", "N05001"],
}
RAG = {
    ("open", "en", "disease"): ["N04001", "N04002", "N04031", "N04064"],
    ("open", "en", "pest"): ["N05001", "N05016"],
    ("open", "zh", "disease"): ["N04021", "N04069", "N04087", "N04126"],
    ("open", "zh", "pest"): ["N05004", "N05014", "N05026", "N05041", "N05042", "N05044", "N05055", "N05058"],
    ("option", "en", "pest"): ["N05001", "N05069"],
}
ATTEMPTS_PER_CLASS = 3


def fresh(code: str, forbidden: set[str], used: set[str], count: int) -> list[tuple[str, str]]:
    selected = []
    directory = IMAGE_ROOT / code
    for path in sorted(directory.glob("*")) if directory.is_dir() else []:
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = str(path.relative_to(ROOT)); digest = image_digest(image, ROOT)
        if digest in forbidden or digest in used:
            continue
        selected.append((image, digest)); used.add(digest)
        if len(selected) == count:
            return selected
    raise ValueError(f"insufficient fresh images for {code}: {len(selected)} < {count}")


def direct(code: str, image: str, digest: str, language: str, question: str, domain: str, names: dict[str, str], classes: dict[str, dict[str, str]], index: int) -> dict[str, Any]:
    target = {
        "target_id": f"hermes-v2-large010-direct-{question}-{language}-{domain}-{index:03d}",
        "query_image": image, "image_sha256": digest, "canonical_class": code,
        "canonical_name": names["chinese_name"] if language == "zh" else names["english_name"],
        "task_domain": domain, "question_type": question, "language": language,
        "paired_image_id": digest, "generation_route": "direct_visual_comparison",
        "label_visible_to_teacher": True,
        "uncertainty": "Uncertainty/不确定性: The image evidence is insufficient for a confidence estimate.",
        "teacher_requirements": {"visible_candidates_min": 4, "neighbor_exclusions_min": 3, "preserve_long_comparison": True},
    }
    # Letter is intentionally balanced across the small repair; the selector
    # still verifies all A/B/C/D floors in the final 70-row view.
    if question == "option":
        target["correct_option"] = "ABCD"[(index - 1) % 4]
        target["candidate_labels"] = labels(classes, code, domain, target["correct_option"])
    return target


def rag(code: str, image: str, digest: str, question: str, language: str, domain: str, names: dict[str, str], classes: dict[str, dict[str, str]], index: int) -> dict[str, Any]:
    target = {
        "target_id": f"hermes-v2-large010-rag-{question}-{language}-{domain}-{index:03d}",
        "query_image": image, "image_sha256": digest, "canonical_class": code,
        "canonical_name": names["chinese_name"] if language == "zh" else names["english_name"],
        "task_domain": domain, "question_type": question, "language": language,
        "generation_route": "blind_evidence", "label_visible_to_teacher": False,
        "uncertainty": "Uncertainty/不确定性: The image evidence is insufficient for a confidence estimate.",
        "teacher_requirements": {"pre_tool_visual_candidates_min": 3, "evidence_exclusions_min": 2, "public_tool_evidence_only": True},
    }
    if question == "option":
        target["correct_option"] = "ABCD"[(index - 1) % 4]
        target["candidate_labels"] = labels(classes, code, domain, target["correct_option"])
    return target


def main() -> int:
    if DEST.exists():
        raise RuntimeError(f"destination exists: {DEST}")
    classes = catalog()
    forbidden = set(json.loads(ISOLATION.read_text(encoding="utf-8"))) | contacted_hashes(COLLECTION)
    used: set[str] = set(); direct_rows: list[dict[str, Any]] = []; rag_rows: list[dict[str, Any]] = []
    for (question, domain), codes in DIRECT.items():
        index = 0
        for code in codes:
            for image, digest in fresh(code, forbidden, used, ATTEMPTS_PER_CLASS):
                index += 1
                for language in ("en", "zh"):
                    direct_rows.append(direct(code, image, digest, language, question, domain, classes[code], classes, index))
    for (question, language, domain), codes in RAG.items():
        index = 0
        for code in codes:
            for image, digest in fresh(code, forbidden, used, ATTEMPTS_PER_CLASS):
                index += 1
                rag_rows.append(rag(code, image, digest, question, language, domain, classes[code], classes, index))
    if {x["image_sha256"] for x in direct_rows} & {x["image_sha256"] for x in rag_rows}:
        raise ValueError("direct/rag image overlap")
    DEST.mkdir(parents=True)
    for route, rows in (("direct", direct_rows), ("rag", rag_rows)):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(f"{route}-{row['question_type']}-{row['language']}-{row['task_domain']}", []).append(row)
        for key, group in grouped.items():
            write_jsonl(DEST / f"{key}.jsonl", group)
    report = {
        "schema_version": "agrinet.hermes-large010-final-quota-supplement/v1",
        "forbidden_hashes": len(forbidden), "direct_rows": len(direct_rows), "rag_rows": len(rag_rows),
        "unique_images": len(used), "direct_sha256": canonical_json_hash(direct_rows),
        "rag_sha256": canonical_json_hash(rag_rows), "training_authorized": False,
    }
    (DEST / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
