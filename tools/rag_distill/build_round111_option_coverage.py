#!/usr/bin/env python3
"""Build the four-row Round111 Blind Option coverage Pilot plan.

The builder is static-only: it never calls the teacher or Milvus.  Every query
image is excluded if it has appeared in an earlier plan, raw trajectory,
accepted corpus, freeze, or evaluation manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.rag_distill.build_round098_option_coverage import (
    ROOT,
    audited_catalog,
    candidate_labels,
    evaluation_images,
    normalized,
    read_jsonl,
)

ROUND = 111
TEACHER_TEMPERATURE = 0.0
OUT = ROOT / "outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round111_option_coverage"
ARTIFACT = ROOT / "outputs/experiments/rag_sft_iteration/candidates/round111_option_coverage"

# Four distinct classes with prior real-Milvus target-recall evidence.  The
# letters, languages, and domains are balanced while image instances remain new.
SPECS = (
    ("N04106", "en", "disease", "A"),
    ("N04071", "zh", "disease", "B"),
    ("N05013", "en", "pest", "C"),
    ("N05010", "zh", "pest", "D"),
)


def sample_image(row: dict[str, Any]) -> str | None:
    sample = row.get("sample") or row.get("trace", {}).get("sample") or row
    metadata = row.get("metadata") or {}
    images = row.get("images") or []
    return normalized(sample.get("query_image") or metadata.get("query_image") or (images[0] if images else None))


def exposed_images() -> set[str]:
    images = set(evaluation_images())
    plans = ROOT / "outputs/experiments/rag_sft_iteration/rounds"
    for path in plans.rglob("plan.jsonl"):
        for row in read_jsonl(path):
            image = sample_image(row)
            if image:
                images.add(image)
    candidates = ROOT / "outputs/experiments/rag_sft_iteration/candidates"
    for path in candidates.rglob("*.jsonl"):
        if path.name not in {"raw_trajectories.jsonl", "agent_sft.accepted.jsonl"}:
            continue
        for row in read_jsonl(path):
            image = sample_image(row)
            if image:
                images.add(image)
    artifacts = ROOT / "outputs/artifacts/datasets"
    for path in artifacts.glob("agrinet-rag-*/data.jsonl"):
        for row in read_jsonl(path):
            image = sample_image(row)
            if image:
                images.add(image)
    return images


def choose_fresh_image(code: str, forbidden: set[str]) -> str:
    folder = ROOT / "datasets/AgriNet-1K/all" / code
    for path in sorted(folder.glob(f"{code}_P*.jpg")):
        image = str(path.relative_to(ROOT))
        if image not in forbidden:
            return image
    raise RuntimeError(f"no fresh Round111 image for {code}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def main() -> None:
    classes = audited_catalog()
    forbidden = exposed_images()
    excluded_before_selection = len(forbidden)
    source_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []

    for index, (code, language, domain, letter) in enumerate(SPECS, start=1):
        info = classes.get(code)
        if not info or info["task_domain"] != domain:
            raise RuntimeError(f"missing or mismatched audited metadata for {code}")
        image = choose_fresh_image(code, forbidden)
        labels = candidate_labels(classes, code, domain, letter)
        if len(labels) != 4 or labels[ord(letter) - ord("A")]["code"] != code:
            raise RuntimeError(f"invalid public Option mapping for {code}")
        source_id = f"round111_{language}_{domain}_{code}_{Path(image).stem}"
        source = {
            "sample_id": source_id,
            "source_sample_id": source_id,
            "target_id": f"rag_option-round111-{source_id}",
            "query_image": image,
            "image_sha256": hashlib.sha256((ROOT / image).read_bytes()).hexdigest(),
            "class_code": code,
            "class_name": info["english_name"],
            "class_name_zh": info["chinese_name"],
            "task_domain": domain,
            "language": language,
            "question_type": "option",
            "candidate_labels": labels,
            "final_label": code,
            "final_label_zh": info["chinese_name"],
            "correct_option": letter,
            "trajectory_mode": "standard",
            "train_eligible": True,
            "max_tool_turns": 3,
            "generation_route": "blind_evidence",
            "label_visible_to_teacher": False,
            "strategy_id": "visual_then_balanced",
            "preferred_sequence": ["visual", "balanced"],
            "retrieval_top_k": 5,
            "top_k": 5,
            "round": ROUND,
            "focus": ["fresh_image", "option_abcd_coverage", "blind_evidence", "language_isolation"],
            "teacher_temperature": TEACHER_TEMPERATURE,
            "preflight_required": True,
            "preflight_eligible": None,
        }
        source_rows.append(source)
        plan_rows.append(dict(source, sample_id=f"round111-option-{index}-{source_id}"))
        forbidden.add(image)

    expected = {"letters": {"A": 1, "B": 1, "C": 1, "D": 1}, "languages": {"en": 2, "zh": 2}, "domains": {"disease": 2, "pest": 2}}
    actual = {
        "letters": dict(sorted(Counter(row["correct_option"] for row in plan_rows).items())),
        "languages": dict(sorted(Counter(row["language"] for row in plan_rows).items())),
        "domains": dict(sorted(Counter(row["task_domain"] for row in plan_rows).items())),
    }
    if len(plan_rows) != 4 or len({row["query_image"] for row in plan_rows}) != 4 or actual != expected:
        raise RuntimeError(json.dumps({"rows": len(plan_rows), "coverage": actual}, ensure_ascii=False))

    report = {
        "round": ROUND,
        "status": "static_audit_passed_preflight_required",
        "rows": len(plan_rows),
        "question_type": "option",
        "route": "blind_evidence",
        "coverage": actual,
        "unique_classes": len({row["class_code"] for row in plan_rows}),
        "unique_query_images": len({row["query_image"] for row in plan_rows}),
        "historical_exposed_and_evaluation_images_excluded": excluded_before_selection,
        "teacher_temperature": TEACHER_TEMPERATURE,
        "training_authorized": False,
        "formal_eval_authorized": False,
        "next_gate": "offline_ablation_then_one_strict_preflight_then_bounded_four_row_pilot",
    }
    for directory in (OUT, ARTIFACT):
        directory.mkdir(parents=True, exist_ok=True)
        write_jsonl(directory / "source.jsonl", source_rows)
        write_jsonl(directory / "plan.jsonl", plan_rows)
        (directory / "static_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(OUT / "plan.jsonl"), "audit": report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
