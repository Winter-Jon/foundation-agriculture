#!/usr/bin/env python3
"""Fresh Blind-only repair for the final v6 RAG quota gaps."""
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
DEST = ROOT / "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-20260819-large-013/supplement"
IMAGE_ROOT = ROOT / "datasets/AgriNet-1K/all"

# v6: open/zh/disease is short one row and open/zh/pest two rows.  These
# classes have previously passed the same Blind protocol but are absent from
# their respective selected cells, so they preserve class diversity.
TARGETS = {
    ("open", "zh", "disease"): ["N04090"],
    ("open", "zh", "pest"): ["N05007", "N05030"],
}
ATTEMPTS_PER_CLASS = 4


def fresh_images(code: str, forbidden: set[str], used: set[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for path in sorted((IMAGE_ROOT / code).glob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = str(path.relative_to(ROOT))
        digest = image_digest(image, ROOT)
        if digest in forbidden or digest in used:
            continue
        rows.append((image, digest)); used.add(digest)
        if len(rows) == ATTEMPTS_PER_CLASS:
            return rows
    raise ValueError(f"insufficient fresh images for {code}")


def main() -> int:
    if DEST.exists():
        raise RuntimeError(f"destination exists: {DEST}")
    classes = catalog()
    forbidden = set(json.loads(ISOLATION.read_text(encoding="utf-8"))) | contacted_hashes(COLLECTION)
    used: set[str] = set(); rows: list[dict[str, object]] = []
    for (question, language, domain), codes in TARGETS.items():
        index = 0
        for code in codes:
            for image, digest in fresh_images(code, forbidden, used):
                index += 1; entry = classes[code]
                rows.append({
                    "target_id": f"hermes-v2-large013-rag-{question}-{language}-{domain}-{index:03d}",
                    "query_image": image, "image_sha256": digest, "canonical_class": code,
                    "canonical_name": entry["chinese_name"], "task_domain": domain,
                    "question_type": question, "language": language,
                    "generation_route": "blind_evidence", "label_visible_to_teacher": False,
                    "uncertainty": "Uncertainty/不确定性: The image evidence is insufficient for a confidence estimate.",
                    "teacher_requirements": {"pre_tool_visual_candidates_min": 3, "evidence_exclusions_min": 2, "public_tool_evidence_only": True},
                })
    DEST.mkdir(parents=True)
    for cell in TARGETS:
        question, language, domain = cell
        write_jsonl(DEST / f"rag-{question}-{language}-{domain}.jsonl", [r for r in rows if (r["question_type"], r["language"], r["task_domain"]) == cell])
    report = {"schema_version": "agrinet.hermes-large013-final-blind-quota/v1", "rows": len(rows), "sha256": canonical_json_hash(rows), "forbidden_hashes": len(forbidden), "training_authorized": False}
    (DEST / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
