"""Record conservative decisions for latent correction candidates."""
import json
from pathlib import Path

src=Path("outputs/experiments/rag_sft_iteration/reviews/latent_correction_recommendations.jsonl")
out=Path("outputs/experiments/rag_sft_iteration/reviews/latent_correction_decisions.jsonl")
rows=[]
for line in src.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    r=json.loads(line)
    if r.get("recommendation")=="latent_narrowed_review":
        r["decision"]="do_not_relabel_existing_row"
        r["future_use"]="prompt_calibration_only"
        r["training_eligible_now"]=False
        r["decision_reason"]="historical trace lacks generation-time desired_correction_type provenance; replay would duplicate an existing image"
        rows.append(r)
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows),encoding="utf-8")
print(json.dumps({"decisions":len(rows),"training_eligible_now":0,"policy":"immutable historical rows"},ensure_ascii=False))
