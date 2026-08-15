"""Build a read-only review package for historical correction semantics."""
from __future__ import annotations
import json, glob, re
from pathlib import Path

out = Path("outputs/experiments/rag_sft_iteration/reviews/latent_correction_candidates.jsonl")
rows=[]
for path in glob.glob("outputs/experiments/rag_sft_iteration/candidates/*/train/agent_sft.accepted.jsonl"):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row=json.loads(line); meta=row.get("metadata",{}); audit=meta.get("correction_audit") or {}
        if audit.get("change_type") != "class_changed": continue
        assistants=[str(m.get("content", "")) for m in row.get("messages",[]) if m.get("role")=="assistant"]
        text="\n".join(assistants)
        if not any(t in text.lower() for t in ("narrow", "refined", "specific", "收窄", "具体", "排除", "候选")): continue
        candidates=[]
        for marker in ("Rejected alternatives:", "排除的候选:"):
            if marker in text:
                block=text.split(marker,1)[1].split("Uncertainty:",1)[0].split("不确定性:",1)[0]
                candidates.extend(re.findall(r"[-•]\s*([^\n]{3,180})",block))
        rows.append({"sample_id":row.get("sample_id"),"source_artifact":path,"language":meta.get("language"),"domain":meta.get("task_domain"),"original_change_type":audit.get("change_type"),"initial_hypothesis":audit.get("initial_hypothesis"),"revised_hypothesis":audit.get("revised_hypothesis"),"candidate_exclusions":candidates[:6],"review_label":"needs_human_semantic_review"})
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows),encoding="utf-8")
summary={"rows":len(rows),"review_label":"read_only_no_training_relabel","source_count":len({r["source_artifact"] for r in rows})}
(out.with_suffix(".summary.json")).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False))
