from pathlib import Path
import glob, json
from collections import Counter

rows = []
for fn in glob.glob("outputs/experiments/rag_sft_iteration/candidates/**/train/agent_sft.accepted.jsonl", recursive=True):
    if "/round-" in fn:
        continue
    for line in Path(fn).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        x = json.loads(line)
        m = x.get("metadata") or {}
        imgs = x.get("images") or []
        sid = str(x.get("sample_id"))
        rows.append({"sample_id": x.get("sample_id"), "source": str(Path(fn).parents[4]), "language": m.get("language"), "domain": m.get("task_domain") or ("pest" if sid.startswith("pest_") else "disease" if sid.startswith("disease_") else None), "question_type": m.get("question_type"), "route": m.get("generation_route"), "strategy": m.get("strategy_id"), "retrieval_types": m.get("retrieval_types"), "label_visible": m.get("label_visible_to_teacher"), "images": imgs, "correction_type": (m.get("correction_audit") or {}).get("change_type")})
ids = [r["sample_id"] for r in rows]
imgs = [r["images"][0] for r in rows if r["images"]]
counts = {k: dict(Counter(r[k] for r in rows)) for k in ("language", "domain", "question_type", "route", "correction_type")}
report = {"accepted": len(rows), "rows": rows, "counts": counts, "duplicates": {"sample_id": len(ids)-len(set(ids)), "target_image": len(imgs)-len(set(imgs))}, "gate": {"minimum_rows": 16, "minimum_languages": 2, "minimum_domains": 2, "minimum_question_types": 2, "minimum_routes": 2}, "decision": "no_freeze_no_sft" if len(rows) < 16 else "review_gate"}
out = Path("outputs/experiments/rag_sft_iteration/reviews/all_accepted_coverage_audit.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "accepted": len(rows), "counts": counts, "duplicates": report["duplicates"], "decision": report["decision"]}, ensure_ascii=False))
