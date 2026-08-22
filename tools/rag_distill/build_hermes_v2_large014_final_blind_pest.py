#!/usr/bin/env python3
"""One-cell fresh Blind RAG repair for v7's final open/zh/pest row."""
from __future__ import annotations

import json
from pathlib import Path

from agrinet.data.rebuild_sft import canonical_json_hash, image_digest
from agrinet.data.sft_recovery import write_jsonl
from tools.rag_distill.build_hermes_1to1_collection_shard import contacted_hashes
from tools.rag_distill.build_hermes_v2_large009_targeted_supplement import catalog

ROOT = Path(__file__).resolve().parents[2]
COLLECTION = ROOT / "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection"
ISOLATION = ROOT / "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/isolation/forbidden_image_sha256.json"
DEST = ROOT / "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-20260819-large-014/supplement"
IMAGE_ROOT = ROOT / "datasets/AgriNet-1K/all"
CODE = "N05050"
ATTEMPTS = 6


def main() -> int:
    if DEST.exists():
        raise RuntimeError(f"destination exists: {DEST}")
    forbidden = set(json.loads(ISOLATION.read_text(encoding="utf-8"))) | contacted_hashes(COLLECTION)
    rows = []
    for path in sorted((IMAGE_ROOT / CODE).glob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = str(path.relative_to(ROOT)); digest = image_digest(image, ROOT)
        if digest in forbidden:
            continue
        rows.append({
            "target_id": f"hermes-v2-large014-rag-open-zh-pest-{len(rows)+1:03d}",
            "query_image": image, "image_sha256": digest, "canonical_class": CODE,
            "canonical_name": catalog()[CODE]["chinese_name"], "task_domain": "pest",
            "question_type": "open", "language": "zh",
            "generation_route": "blind_evidence", "label_visible_to_teacher": False,
            "uncertainty": "Uncertainty/不确定性: The image evidence is insufficient for a confidence estimate.",
            "teacher_requirements": {"pre_tool_visual_candidates_min": 3, "evidence_exclusions_min": 2, "public_tool_evidence_only": True},
        })
        if len(rows) == ATTEMPTS:
            break
    if len(rows) != ATTEMPTS:
        raise ValueError(f"insufficient fresh images for {CODE}: {len(rows)}")
    DEST.mkdir(parents=True); write_jsonl(DEST / "rag-open-zh-pest.jsonl", rows)
    report = {"schema_version": "agrinet.hermes-large014-final-blind-pest/v1", "rows": len(rows), "sha256": canonical_json_hash(rows), "forbidden_hashes": len(forbidden), "training_authorized": False}
    (DEST / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
