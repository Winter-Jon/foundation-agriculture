#!/usr/bin/env python3
"""Build one fresh Blind Option smoke row for the repaired answer contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from archive.source.rag_distill.archive.round_builders.build_round098_option_coverage import (
    ROOT,
    audited_catalog,
    candidate_labels,
    evaluation_images,
    normalized,
    read_jsonl,
)

ROUND = int(os.environ.get("RAG_SFT_OPTION_SMOKE_ROUND", "99"))
OUT = ROOT / f"outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round{ROUND:03d}_option_contract_smoke"
ARTIFACT = ROOT / f"outputs/experiments/rag_sft_iteration/candidates/round{ROUND:03d}_option_contract_smoke"
CODE, LANGUAGE, DOMAIN, LETTER = "N04062", "en", "disease", "D"
TEACHER_TEMPERATURE = 0.0


def forbidden_images() -> set[str]:
    forbidden = evaluation_images()
    # A preflight sends the query image to the teacher even when no Pilot
    # artifact is created. Exclude every prior Option-smoke plan so later
    # rounds remain fresh across both preflight and Pilot exposure.
    smoke_plans = ROOT / "outputs/experiments/rag_sft_iteration/rounds/round_0001/plan"
    for path in smoke_plans.glob("round*_option_contract_smoke/plan.jsonl"):
        for row in read_jsonl(path):
            image = normalized(row.get("query_image"))
            if image:
                forbidden.add(image)
    for path in (ROOT / "outputs/experiments/rag_sft_iteration/candidates").rglob("*.jsonl"):
        if path.name not in {"agent_sft.accepted.jsonl", "raw_trajectories.jsonl"}:
            continue
        for row in read_jsonl(path):
            sample = row.get("sample") or row.get("trace", {}).get("sample") or row
            metadata = row.get("metadata", {})
            image = normalized(sample.get("query_image") or metadata.get("query_image") or (row.get("images") or [None])[0])
            if image:
                forbidden.add(image)
    freeze = ROOT / "outputs/artifacts/datasets/agrinet-rag-sft-round097-candidate-freeze/data.jsonl"
    for row in read_jsonl(freeze):
        image = normalized(row.get("metadata", {}).get("query_image") or (row.get("images") or [None])[0])
        if image:
            forbidden.add(image)
    return forbidden


def main() -> None:
    classes = audited_catalog()
    forbidden = forbidden_images()
    folder = ROOT / "datasets/AgriNet-1K/all" / CODE
    image = next((str(path.relative_to(ROOT)) for path in sorted(folder.glob(f"{CODE}_P*.jpg")) if str(path.relative_to(ROOT)) not in forbidden), None)
    if not image:
        raise RuntimeError(f"no fresh Option-contract smoke image for {CODE}")
    labels = candidate_labels(classes, CODE, DOMAIN, LETTER)
    if len(labels) != 4 or labels[ord(LETTER) - ord("A")]["code"] != CODE:
        raise RuntimeError("Option contract smoke construction failed")
    source_id = f"round{ROUND:03d}_{LANGUAGE}_{DOMAIN}_{CODE}_{Path(image).stem}"
    source = {
        "sample_id": source_id, "source_sample_id": source_id,
        "target_id": f"rag_option-round{ROUND:03d}-{source_id}", "query_image": image,
        "image_sha256": hashlib.sha256((ROOT / image).read_bytes()).hexdigest(),
        "class_code": CODE, "class_name": classes[CODE]["english_name"],
        "class_name_zh": classes[CODE]["chinese_name"], "task_domain": DOMAIN,
        "language": LANGUAGE, "question_type": "option", "candidate_labels": labels,
        "final_label": CODE, "final_label_zh": classes[CODE]["chinese_name"],
        "correct_option": LETTER, "trajectory_mode": "standard", "train_eligible": False,
        "max_tool_turns": 3, "generation_route": "blind_evidence",
        "label_visible_to_teacher": False, "strategy_id": "visual_then_balanced",
        "preferred_sequence": ["visual", "balanced"], "retrieval_top_k": 5, "top_k": 5,
        "round": ROUND, "focus": ["option_letter_contract_smoke", "fresh_image", "blind_evidence"],
        "teacher_temperature": TEACHER_TEMPERATURE,
        "preflight_required": True, "preflight_eligible": None,
    }
    plan = dict(source, sample_id=f"round{ROUND:03d}-option-smoke-{source_id}")
    for directory in (OUT, ARTIFACT):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "source.jsonl").write_text(json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8")
        (directory / "plan.jsonl").write_text(json.dumps(plan, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "round": ROUND, "status": "static_audit_passed_preflight_required", "rows": 1,
        "query_image": image, "correct_option": LETTER, "route": "blind_evidence",
        "excluded_images": len(forbidden), "training_authorized": False,
        "formal_eval_authorized": False, "purpose": "Option answer-letter contract smoke only",
        "teacher_temperature": TEACHER_TEMPERATURE,
    }
    for directory in (OUT, ARTIFACT):
        (directory / "static_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
