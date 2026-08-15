#!/usr/bin/env python3
"""Build a fresh, static-audited Option coverage plan for Round098.

This builder deliberately does not call a teacher or Milvus.  It only selects
previously unused query images and produces the source/plan/audit package that
must pass teacher preflight before a bounded Blind pilot may run.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ROOT / "outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl"
OUTPUT = ROOT / "outputs/experiments/rag_sft_iteration/candidates/round098_option_coverage"
PLAN_OUTPUT = ROOT / "outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round098_option_coverage"

# Each requested answer letter is represented twice across language/domain when
# possible.  The run remains bounded: this is a six-row preflight package, not
# authorization for a larger teacher batch.
SPECS = (
    ("N04007", "en", "disease", "A"),
    ("N04029", "zh", "disease", "C"),
    ("N05013", "en", "pest", "C"),
    ("N05045", "zh", "pest", "D"),
    ("N04062", "en", "disease", "D"),
    ("N05027", "zh", "pest", "A"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalized(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return str(candidate.relative_to(ROOT))
        except ValueError:
            return str(candidate)
    return str(candidate)


def accepted_images() -> set[str]:
    images: set[str] = set()
    candidates = ROOT / "outputs/experiments/rag_sft_iteration/candidates"
    for path in candidates.glob("*/train/agent_sft.accepted.jsonl"):
        for row in read_jsonl(path):
            metadata = row.get("metadata", {})
            image = metadata.get("query_image") or (row.get("images") or [None])[0]
            image = normalized(image)
            if image:
                images.add(image)
    freeze = ROOT / "outputs/artifacts/datasets/agrinet-rag-sft-round097-candidate-freeze/data.jsonl"
    if freeze.exists():
        for row in read_jsonl(freeze):
            metadata = row.get("metadata", {})
            image = normalized(metadata.get("query_image") or (row.get("images") or [None])[0])
            if image:
                images.add(image)
    return images


def evaluation_images() -> set[str]:
    images: set[str] = set()
    for path in (ROOT / "outputs").rglob("manifest.jsonl"):
        # Current evaluation manifests live both under `vlm_eval` and under
        # iteration-control `.../evaluation/...`; do not infer isolation from
        # a directory name that happens not to equal exactly `eval`.
        if "vlm_eval" not in path.parts and "evaluation" not in path.parts:
            continue
        for row in read_jsonl(path):
            image = normalized(row.get("image_path") or row.get("query_image"))
            if image:
                images.add(image)
    return images


def choose_image(code: str, forbidden: set[str]) -> str:
    folder = ROOT / "datasets/AgriNet-1K/all" / code
    for image in sorted(folder.glob(f"{code}_P*.jpg")):
        relative = str(image.relative_to(ROOT))
        if relative not in forbidden:
            return relative
    raise RuntimeError(f"no unused image available for {code}")


def audited_catalog() -> dict[str, dict[str, str]]:
    """Build class metadata from compact class data plus audited historic plans.

    The bounded contrast artifact only has eight classes, while the legacy
    Option plans contain the additional agriculture classes.  We reuse names
    and domains only; query images are still selected fresh below.
    """
    catalog: dict[str, dict[str, str]] = {}
    for entry in read_jsonl(CLASSES):
        catalog[entry["code"]] = {
            "code": entry["code"],
            "english_name": entry["english_name"],
            "chinese_name": entry["chinese_name"],
            "task_domain": entry["task_domain"],
        }
    plans = ROOT / "outputs/experiments/rag_sft_iteration/rounds/round_0001/plan"
    for path in plans.rglob("*.jsonl"):
        for row in read_jsonl(path):
            code = row.get("class_code")
            if code and row.get("class_name") and row.get("class_name_zh") and row.get("task_domain"):
                catalog.setdefault(code, {
                    "code": code,
                    "english_name": row["class_name"],
                    "chinese_name": row["class_name_zh"],
                    "task_domain": row["task_domain"],
                })
            for label in row.get("candidate_labels") or []:
                label_code = label.get("code")
                if not label_code:
                    continue
                english = label.get("name") or label.get("english_name")
                chinese = label.get("chinese_name")
                domain = label.get("task_domain")
                if english and chinese and domain:
                    catalog.setdefault(label_code, {
                        "code": label_code,
                        "english_name": english,
                        "chinese_name": chinese,
                        "task_domain": domain,
                    })
    return catalog


def candidate_labels(classes: dict[str, dict[str, str]], code: str, domain: str, correct_option: str) -> list[dict[str, str]]:
    pool = [entry for entry in classes.values() if entry["task_domain"] == domain and entry["code"] != code]
    pool.sort(key=lambda entry: entry["code"])
    distractors = pool[:3]
    target = classes[code]
    labels = distractors[:]
    labels.insert(ord(correct_option) - ord("A"), target)
    return [
        {
            "code": entry["code"],
            "name": entry["english_name"],
            "chinese_name": entry["chinese_name"],
            "task_domain": entry["task_domain"],
        }
        for entry in labels
    ]


def main() -> None:
    classes = audited_catalog()
    used = accepted_images()
    eval_used = evaluation_images()
    forbidden = used | eval_used
    source_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    static_rejections: list[dict[str, str]] = []

    for index, (code, language, domain, correct_option) in enumerate(SPECS, start=1):
        if code not in classes:
            raise RuntimeError(f"missing audited class metadata for {code}")
        if classes[code]["task_domain"] != domain:
            raise RuntimeError(f"domain mismatch for {code}: {classes[code]['task_domain']} != {domain}")
        try:
            image = choose_image(code, forbidden)
        except RuntimeError as exc:
            static_rejections.append({"code": code, "reason": str(exc)})
            continue
        info = classes[code]
        labels = candidate_labels(classes, code, domain, correct_option)
        if len(labels) != 4 or labels[ord(correct_option) - ord("A")]["code"] != code:
            raise RuntimeError(f"invalid Option contract for {code}")
        source_id = f"round098_{language}_{domain}_{code}_{Path(image).stem}"
        source_row = {
            "sample_id": source_id,
            "source_sample_id": source_id,
            "target_id": f"rag_option-round098-{source_id}",
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
            "correct_option": correct_option,
            "trajectory_mode": "standard",
            "train_eligible": True,
            "max_tool_turns": 3,
            "generation_route": "blind_evidence",
            "label_visible_to_teacher": False,
            "strategy_id": "visual_then_balanced",
            "preferred_sequence": ["visual", "balanced"],
            "retrieval_top_k": 5,
            "top_k": 5,
            "round": 98,
            "focus": ["fresh_image", "option_contract", "blind_evidence", "language_isolation", "actual_visual_recall"],
            "preflight_required": True,
            "preflight_eligible": None,
        }
        plan_row = dict(source_row)
        plan_row["sample_id"] = f"round098-option-{index}-{source_id}"
        source_rows.append(source_row)
        plan_rows.append(plan_row)
        forbidden.add(image)

    if static_rejections or len(plan_rows) != len(SPECS):
        raise RuntimeError(json.dumps({"static_rejections": static_rejections, "built": len(plan_rows)}, ensure_ascii=False))
    if len({row["query_image"] for row in plan_rows}) != len(plan_rows):
        raise RuntimeError("duplicate query image in Round098 plan")

    for directory in (OUTPUT, PLAN_OUTPUT):
        directory.mkdir(parents=True, exist_ok=True)
    source_payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in source_rows)
    plan_payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in plan_rows)
    (OUTPUT / "source.jsonl").write_text(source_payload, encoding="utf-8")
    (OUTPUT / "plan.jsonl").write_text(plan_payload, encoding="utf-8")
    (PLAN_OUTPUT / "source.jsonl").write_text(source_payload, encoding="utf-8")
    (PLAN_OUTPUT / "plan.jsonl").write_text(plan_payload, encoding="utf-8")
    audit = {
        "round": 98,
        "status": "static_audit_passed_preflight_required",
        "rows": len(plan_rows),
        "question_type": "option",
        "route": "blind_evidence",
        "answer_distribution": dict(sorted(Counter(row["correct_option"] for row in plan_rows).items())),
        "languages": dict(sorted(Counter(row["language"] for row in plan_rows).items())),
        "domains": dict(sorted(Counter(row["task_domain"] for row in plan_rows).items())),
        "historical_accepted_images_excluded": len(used),
        "evaluation_images_excluded": len(eval_used),
        "training_authorized": False,
        "formal_eval_authorized": False,
        "next_gate": "teacher_preflight_then_bounded_blind_pilot",
    }
    for directory in (OUTPUT, PLAN_OUTPUT):
        (directory / "static_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(PLAN_OUTPUT / "plan.jsonl"), "audit": audit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
