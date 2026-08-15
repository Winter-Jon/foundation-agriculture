"""Summarize latent narrowed candidates without changing training artifacts."""
import json
from pathlib import Path

src=Path("outputs/experiments/rag_sft_iteration/reviews/latent_correction_candidates.jsonl")
out=Path("outputs/experiments/rag_sft_iteration/reviews/latent_correction_dedup_summary.json")
rows=[json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
by={}
for r in rows:
    key=(r.get("sample_id"),r.get("language"),r.get("domain"))
    if key not in by: by[key]=r
strong=[]
for r in by.values():
    text=(r.get("revised_hypothesis") or "").lower()
    if len(r.get("candidate_exclusions") or [])>=1 and any(w in text for w in ("refined","narrow","specific","收窄","具体")):
        strong.append(r)
summary={"source_rows":len(rows),"deduplicated_rows":len(by),"strong_latent_candidates":len(strong),"strong_by_language_domain":{},"policy":"read_only; original class_changed labels and training rows unchanged","candidate_ids":[r.get("sample_id") for r in strong]}
for r in strong:
    k=f"{r.get('language')}/{r.get('domain')}"; summary["strong_by_language_domain"][k]=summary["strong_by_language_domain"].get(k,0)+1
out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False))
