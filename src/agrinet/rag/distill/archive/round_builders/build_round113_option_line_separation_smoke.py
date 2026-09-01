#!/usr/bin/env python3
"""Build one fresh C Option row for line-separated final fields."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agrinet.rag.distill.archive.round_builders.build_round098_option_coverage import ROOT, audited_catalog, candidate_labels
from agrinet.rag.distill.archive.round_builders.build_round111_option_coverage import exposed_images, write_jsonl

ROUND = 113
CODE, LANGUAGE, DOMAIN, LETTER = "N05013", "en", "pest", "C"
TEACHER_TEMPERATURE = 0.0
OUT = ROOT / "outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round113_option_line_separation_smoke"
ARTIFACT = ROOT / "outputs/experiments/rag_sft_iteration/candidates/round113_option_line_separation_smoke"


def main() -> None:
    classes = audited_catalog()
    forbidden = exposed_images()
    folder = ROOT / "datasets/AgriNet-1K/all" / CODE
    image = next((str(p.relative_to(ROOT)) for p in sorted(folder.glob(f"{CODE}_P*.jpg")) if str(p.relative_to(ROOT)) not in forbidden), None)
    if not image:
        raise RuntimeError(f"no fresh Round113 image for {CODE}")
    info = classes[CODE]
    labels = candidate_labels(classes, CODE, DOMAIN, LETTER)
    if labels[ord(LETTER) - ord("A")]["code"] != CODE:
        raise RuntimeError("invalid Round113 Option mapping")
    source_id = f"round113_{LANGUAGE}_{DOMAIN}_{CODE}_{Path(image).stem}"
    source = {
        "sample_id": source_id, "source_sample_id": source_id,
        "target_id": f"rag_option-round113-{source_id}", "query_image": image,
        "image_sha256": hashlib.sha256((ROOT / image).read_bytes()).hexdigest(),
        "class_code": CODE, "class_name": info["english_name"], "class_name_zh": info["chinese_name"],
        "task_domain": DOMAIN, "language": LANGUAGE, "question_type": "option",
        "candidate_labels": labels, "final_label": CODE, "final_label_zh": info["chinese_name"],
        "correct_option": LETTER, "trajectory_mode": "standard", "train_eligible": True,
        "max_tool_turns": 3, "generation_route": "blind_evidence", "label_visible_to_teacher": False,
        "strategy_id": "visual_then_balanced", "preferred_sequence": ["visual", "balanced"],
        "retrieval_top_k": 5, "top_k": 5, "round": ROUND,
        "focus": ["line_separated_final_fields", "exact_retrieval_anchor", "fresh_image"],
        "teacher_temperature": TEACHER_TEMPERATURE, "preflight_required": True, "preflight_eligible": None,
    }
    plan = dict(source, sample_id=f"round113-option-{source_id}")
    report = {
        "round": ROUND, "status": "static_audit_passed_preflight_required", "rows": 1,
        "letter": LETTER, "domain": DOMAIN, "language": LANGUAGE, "query_image": image,
        "historical_exposed_and_evaluation_images_excluded": len(forbidden),
        "teacher_temperature": TEACHER_TEMPERATURE, "training_authorized": False,
        "formal_eval_authorized": False, "purpose": "C Option line-separated final-field smoke",
    }
    for directory in (OUT, ARTIFACT):
        directory.mkdir(parents=True, exist_ok=True)
        write_jsonl(directory / "source.jsonl", [source])
        write_jsonl(directory / "plan.jsonl", [plan])
        (directory / "static_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(OUT / "plan.jsonl"), "audit": report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
