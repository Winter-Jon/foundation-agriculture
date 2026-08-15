"""Produce conservative recommendations for latent correction review."""
import json
from pathlib import Path

src=Path("outputs/experiments/rag_sft_iteration/reviews/latent_correction_candidates.jsonl")
out=Path("outputs/experiments/rag_sft_iteration/reviews/latent_correction_recommendations.jsonl")
rows=[]; seen=set()
for line in src.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    row=json.loads(line); key=(row.get("sample_id"),row.get("language"),row.get("domain"))
    if key in seen: continue
    seen.add(key)
    initial=(row.get("initial_hypothesis") or "").lower()
    revised=(row.get("revised_hypothesis") or "").lower()
    exclusions=row.get("candidate_exclusions") or []
    broad=any(t in initial for t in ("possible", "candidate", "generic", "宽泛", "可能"))
    specific=any(t in revised for t in ("refined", "specific", "收窄", "具体", "confirmed", "一致"))
    if broad and specific and exclusions:
        row["recommendation"]="latent_narrowed_review"
        row["recommendation_reason"]="broad initial hypothesis + retrieval-specific revision + explicit rejected alternative"
    else:
        row["recommendation"]="retain_class_changed"
        row["recommendation_reason"]="insufficient explicit evidence for a separate narrowed label"
    rows.append(row)
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows),encoding="utf-8")
print(json.dumps({"rows":len(rows),"latent_narrowed_review":sum(r["recommendation"]=="latent_narrowed_review" for r in rows),"retain_class_changed":sum(r["recommendation"]=="retain_class_changed" for r in rows)},ensure_ascii=False))
