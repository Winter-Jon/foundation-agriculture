#!/usr/bin/env python3
"""Build the final fresh Blind RAG repair after v5 preflight."""
from __future__ import annotations

import json
from pathlib import Path

from agrinet.data.rebuild_sft import canonical_json_hash, image_digest
from agrinet.data.sft_recovery import write_jsonl
from archive.source.rag_distill.build_hermes_1to1_collection_shard import contacted_hashes
from archive.source.rag_distill.build_hermes_v2_large009_targeted_supplement import catalog

ROOT = Path(__file__).resolve().parents[3]
COLLECTION = ROOT / "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection"
ISOLATION = ROOT / "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/isolation/forbidden_image_sha256.json"
DEST = ROOT / "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-20260819-large-012/supplement"
IMAGE_ROOT = ROOT / "datasets/AgriNet-1K/all"
# All are v5 singletons.  Three fresh images/class make a strict rejection
# unlikely to starve the final one- and three-row deficits, while the selector
# keeps the two-row class cap.
TARGETS = {
    ("open", "zh", "disease"): ["N04021"],
    ("open", "zh", "pest"): ["N05062", "N05064", "N05066"],
}
ATTEMPTS_PER_CLASS = 3


def images(code: str, forbidden: set[str], used: set[str]) -> list[tuple[str, str]]:
    result = []
    for path in sorted((IMAGE_ROOT / code).glob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = str(path.relative_to(ROOT)); digest = image_digest(image, ROOT)
        if digest not in forbidden and digest not in used:
            result.append((image, digest)); used.add(digest)
            if len(result) == ATTEMPTS_PER_CLASS: return result
    raise ValueError(f"insufficient fresh images for {code}")


def main() -> int:
    if DEST.exists(): raise RuntimeError(f"destination exists: {DEST}")
    classes = catalog(); forbidden = set(json.loads(ISOLATION.read_text(encoding="utf-8"))) | contacted_hashes(COLLECTION); used=set(); rows=[]
    for (question, language, domain), codes in TARGETS.items():
        index=0
        for code in codes:
            for image,digest in images(code,forbidden,used):
                index+=1; entry=classes[code]
                rows.append({
                    "target_id":f"hermes-v2-large012-rag-{question}-{language}-{domain}-{index:03d}",
                    "query_image":image,"image_sha256":digest,"canonical_class":code,
                    "canonical_name":entry["chinese_name"] if language=="zh" else entry["english_name"],
                    "task_domain":domain,"question_type":question,"language":language,
                    "generation_route":"blind_evidence","label_visible_to_teacher":False,
                    "uncertainty":"Uncertainty/不确定性: The image evidence is insufficient for a confidence estimate.",
                    "teacher_requirements":{"pre_tool_visual_candidates_min":3,"evidence_exclusions_min":2,"public_tool_evidence_only":True},
                })
    DEST.mkdir(parents=True)
    for cell in TARGETS:
        q,l,d=cell; write_jsonl(DEST/f"rag-{q}-{l}-{d}.jsonl",[r for r in rows if (r['question_type'],r['language'],r['task_domain'])==cell])
    report={"schema_version":"agrinet.hermes-large012-final-blind-quota/v1","rows":len(rows),"sha256":canonical_json_hash(rows),"forbidden_hashes":len(forbidden),"training_authorized":False}
    (DEST/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))

if __name__ == "__main__": main()
