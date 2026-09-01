from pathlib import Path
import glob, json
from collections import defaultdict, Counter

rows=[]
for fn in glob.glob("outputs/experiments/rag_sft_iteration/candidates/**/train/agent_sft.accepted.jsonl",recursive=True):
    if "/round-" in fn:
        continue
    for line in Path(fn).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
groups=defaultdict(list)
for row in rows:
    imgs=row.get("images") or []
    groups[imgs[0] if imgs else row.get("sample_id")].append(row)
dedup=[]; decisions=[]
for image, group in groups.items():
    if len(group)==1:
        dedup.append(group[0]); continue
    option=[r for r in group if (r.get("metadata") or {}).get("question_type")=="option"]
    keep=option[0] if option else group[0]
    dedup.append(keep)
    decisions.append({"image":image,"candidates":[r.get("sample_id") for r in group],"retained":keep.get("sample_id"),"reason":"prefer option coverage" if option else "deterministic first accepted"})
counts={k:dict(Counter((r.get("metadata") or {}).get(k) for r in dedup)) for k in ("language","task_domain","question_type","generation_route")}
report={"input_rows":len(rows),"dedup_rows":len(dedup),"duplicate_groups":decisions,"counts":counts,"remaining_duplicate_images":0,"decision":"dedup_review_ready_no_sft_until_human_or_gate_review"}
out=Path("outputs/experiments/rag_sft_iteration/reviews/dedup_freeze_review.json")
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
source=Path("outputs/experiments/rag_sft_iteration/reviews/dedup_freeze_source.jsonl")
source.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in dedup))
print(json.dumps({"report":str(out),"source":str(source),"input_rows":len(rows),"dedup_rows":len(dedup),"counts":counts,"duplicates":decisions},ensure_ascii=False))
