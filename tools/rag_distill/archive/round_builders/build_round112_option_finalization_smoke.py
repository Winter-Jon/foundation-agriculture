#!/usr/bin/env python3
"""Build a fresh two-row A/C smoke for the finalization prompt repair."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from tools.rag_distill.archive.round_builders.build_round098_option_coverage import ROOT, audited_catalog, candidate_labels
from tools.rag_distill.archive.round_builders.build_round111_option_coverage import exposed_images, write_jsonl

ROUND = 112
TEACHER_TEMPERATURE = 0.0
OUT = ROOT / "outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round112_option_finalization_smoke"
ARTIFACT = ROOT / "outputs/experiments/rag_sft_iteration/candidates/round112_option_finalization_smoke"
SPECS = (
    ("N04106", "en", "disease", "A"),
    ("N05013", "en", "pest", "C"),
)


def main() -> None:
    classes = audited_catalog()
    forbidden = exposed_images()
    excluded = len(forbidden)
    source_rows = []
    plan_rows = []
    for index, (code, language, domain, letter) in enumerate(SPECS, start=1):
        info = classes[code]
        folder = ROOT / "datasets/AgriNet-1K/all" / code
        image = next((str(p.relative_to(ROOT)) for p in sorted(folder.glob(f"{code}_P*.jpg")) if str(p.relative_to(ROOT)) not in forbidden), None)
        if not image:
            raise RuntimeError(f"no fresh Round112 image for {code}")
        labels = candidate_labels(classes, code, domain, letter)
        if labels[ord(letter) - ord("A")]["code"] != code:
            raise RuntimeError(f"invalid Option mapping for {code}")
        source_id = f"round112_{language}_{domain}_{code}_{Path(image).stem}"
        source = {
            "sample_id": source_id, "source_sample_id": source_id,
            "target_id": f"rag_option-round112-{source_id}", "query_image": image,
            "image_sha256": hashlib.sha256((ROOT / image).read_bytes()).hexdigest(),
            "class_code": code, "class_name": info["english_name"],
            "class_name_zh": info["chinese_name"], "task_domain": domain,
            "language": language, "question_type": "option", "candidate_labels": labels,
            "final_label": code, "final_label_zh": info["chinese_name"],
            "correct_option": letter, "trajectory_mode": "standard", "train_eligible": True,
            "max_tool_turns": 3, "generation_route": "blind_evidence",
            "label_visible_to_teacher": False, "strategy_id": "visual_then_balanced",
            "preferred_sequence": ["visual", "balanced"], "retrieval_top_k": 5, "top_k": 5,
            "round": ROUND, "focus": ["finalization_exact_fields", "exact_retrieval_anchor", "fresh_image"],
            "teacher_temperature": TEACHER_TEMPERATURE, "preflight_required": True, "preflight_eligible": None,
        }
        source_rows.append(source)
        plan_rows.append(dict(source, sample_id=f"round112-option-{index}-{source_id}"))
        forbidden.add(image)
    coverage = dict(sorted(Counter(row["correct_option"] for row in plan_rows).items()))
    if coverage != {"A": 1, "C": 1} or len({row["query_image"] for row in plan_rows}) != 2:
        raise RuntimeError("Round112 coverage or uniqueness failure")
    report = {
        "round": ROUND, "status": "static_audit_passed_preflight_required", "rows": 2,
        "letters": coverage, "domains": {"disease": 1, "pest": 1}, "languages": {"en": 2},
        "unique_query_images": 2, "historical_exposed_and_evaluation_images_excluded": excluded,
        "teacher_temperature": TEACHER_TEMPERATURE, "training_authorized": False,
        "formal_eval_authorized": False, "purpose": "A/C finalization exact-field and exact-evidence-anchor repair smoke",
    }
    for directory in (OUT, ARTIFACT):
        directory.mkdir(parents=True, exist_ok=True)
        write_jsonl(directory / "source.jsonl", source_rows)
        write_jsonl(directory / "plan.jsonl", plan_rows)
        (directory / "static_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(OUT / "plan.jsonl"), "audit": report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
