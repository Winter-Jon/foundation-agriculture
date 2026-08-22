#!/usr/bin/env python3
"""Build the six fresh Blind RAG attempts remaining after v4 preflight."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agrinet.data.rebuild_sft import canonical_json_hash, image_digest
from agrinet.data.sft_recovery import write_jsonl
from tools.rag_distill.build_hermes_1to1_collection_shard import contacted_hashes
from tools.rag_distill.build_hermes_v2_large009_targeted_supplement import catalog
from tools.rag_distill.build_hermes_1to1_large_plan import labels

ROOT = Path(__file__).resolve().parents[2]
COLLECTION = ROOT / "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection"
ISOLATION = ROOT / "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/isolation/forbidden_image_sha256.json"
DEST = ROOT / "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-20260819-large-011/supplement"
IMAGE_ROOT = ROOT / "datasets/AgriNet-1K/all"
TARGETS = {
    ("open", "zh", "disease"): ["N04021", "N04087"],
    ("open", "zh", "pest"): ["N05004", "N05041", "N05055"],
    ("option", "en", "pest"): ["N05069"],
}


def fresh(code: str, forbidden: set[str]) -> tuple[str, str]:
    directory = IMAGE_ROOT / code
    for path in sorted(directory.glob("*")) if directory.is_dir() else []:
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image = str(path.relative_to(ROOT)); digest = image_digest(image, ROOT)
            if digest not in forbidden:
                return image, digest
    raise ValueError(f"no fresh image for {code}")


def main() -> int:
    if DEST.exists(): raise RuntimeError(f"destination exists: {DEST}")
    classes = catalog(); forbidden = set(json.loads(ISOLATION.read_text(encoding="utf-8"))) | contacted_hashes(COLLECTION); rows: list[dict[str, Any]] = []
    for (question, language, domain), codes in TARGETS.items():
        for index, code in enumerate(codes, 1):
            image, digest = fresh(code, forbidden); forbidden.add(digest)
            entry = classes[code]
            row: dict[str, Any] = {
                "target_id": f"hermes-v2-large011-rag-{question}-{language}-{domain}-{index:03d}",
                "query_image": image, "image_sha256": digest, "canonical_class": code,
                "canonical_name": entry["chinese_name"] if language == "zh" else entry["english_name"],
                "task_domain": domain, "question_type": question, "language": language,
                "generation_route": "blind_evidence", "label_visible_to_teacher": False,
                "uncertainty": "Uncertainty/不确定性: The image evidence is insufficient for a confidence estimate.",
                "teacher_requirements": {"pre_tool_visual_candidates_min": 3, "evidence_exclusions_min": 2, "public_tool_evidence_only": True},
            }
            if question == "option":
                row["correct_option"] = "B"
                row["candidate_labels"] = labels(classes, code, domain, "B")
            rows.append(row)
    DEST.mkdir(parents=True)
    for cell in TARGETS:
        q,l,d=cell; group=[r for r in rows if (r["question_type"],r["language"],r["task_domain"])==cell]
        write_jsonl(DEST/f"rag-{q}-{l}-{d}.jsonl", group)
    report={"schema_version":"agrinet.hermes-large011-final-rag-quota/v1","rows":len(rows),"sha256":canonical_json_hash(rows),"forbidden_hashes":len(forbidden)-len(rows),"training_authorized":False}
    (DEST/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))

if __name__ == "__main__": main()
