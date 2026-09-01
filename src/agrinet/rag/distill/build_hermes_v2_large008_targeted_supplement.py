#!/usr/bin/env python3
"""Extract fresh large-008 attempts for the measured freeze-preflight gaps."""
from __future__ import annotations

import json
from pathlib import Path

from agrinet.data.rebuild_sft import canonical_json_hash
from agrinet.data.sft_recovery import read_jsonl, write_jsonl

PLAN = Path("outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-20260819-large-008/plan")
DEST = PLAN.parent / "supplement"
# Direct count is image pairs; RAG count is independent Blind attempts.
DIRECT = {("open", "disease"): 16, ("open", "pest"): 70, ("option", "disease"): 30, ("option", "pest"): 12}
RAG = {("open", "en", "disease"): 40, ("open", "en", "pest"): 100, ("open", "zh", "disease"): 20, ("open", "zh", "pest"): 140, ("option", "en", "pest"): 20, ("option", "zh", "pest"): 20}


def main() -> int:
    if DEST.exists():
        raise RuntimeError(f"destination exists: {DEST}")
    direct_all, rag_all = read_jsonl(PLAN / "direct_targets.jsonl"), read_jsonl(PLAN / "rag_targets.jsonl")
    direct, rag = [], []
    for (question, domain), count in DIRECT.items():
        langs = {}
        for lang in ("en", "zh"):
            rows = [r for r in direct_all if (r.get("question_type"), r.get("language"), r.get("task_domain")) == (question, lang, domain)][:count]
            if len(rows) != count: raise ValueError(f"missing direct {question}/{lang}/{domain}")
            langs[lang] = rows
        if [r["image_sha256"] for r in langs["en"]] != [r["image_sha256"] for r in langs["zh"]]: raise ValueError("lost bilingual pairing")
        direct.extend(langs["en"] + langs["zh"])
    for cell, count in RAG.items():
        rows = [r for r in rag_all if (r.get("question_type"), r.get("language"), r.get("task_domain")) == cell][:count]
        if len(rows) != count: raise ValueError(f"missing rag {'/'.join(cell)}")
        rag.extend(rows)
    if {r["image_sha256"] for r in direct} & {r["image_sha256"] for r in rag}: raise ValueError("direct/rag overlap")
    DEST.mkdir(parents=True)
    for route, rows in (("direct", direct), ("rag", rag)):
        groups = {}
        for row in rows:
            key = f"{route}-{row['question_type']}-{row['language']}-{row['task_domain']}"
            groups.setdefault(key, []).append(row)
        for key, group in groups.items(): write_jsonl(DEST / f"{key}.jsonl", group)
    report = {"source_plan": str(PLAN), "direct_rows": len(direct), "rag_rows": len(rag), "direct_sha256": canonical_json_hash(direct), "rag_sha256": canonical_json_hash(rag), "excluded": "already-quota-sufficient cells", "training_authorized": False}
    (DEST / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))

if __name__ == "__main__": main()
