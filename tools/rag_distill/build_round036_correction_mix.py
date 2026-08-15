"""Build two unused high-margin correction targets with typed objectives."""
from __future__ import annotations
import json
from pathlib import Path
root=Path("outputs/experiments/rag_sft_iteration")
used=set()
for p in (root/"candidates").glob("*/train/agent_sft.accepted.jsonl"):
  for line in p.read_text(encoding="utf-8").splitlines():
    if line.strip():
      x=json.loads(line); used.add(str(x.get("metadata",{}).get("source_sample_id") or x.get("sample_id") or ""))
all_rows=[]
for p in sorted((root/"rounds/round_0001/plan").glob("*.jsonl")):
  for line in p.read_text(encoding="utf-8").splitlines():
    if line.strip(): all_rows.append(json.loads(line))
chosen=[]
for lang,domain,route,desired in (("en","disease","blind_evidence","narrowed"),("zh","pest","oracle_grounded","evidence_confirmed")):
  opts=[]
  for r in all_rows:
    sid=str(r.get("source_sample_id") or r.get("sample_id") or "")
    pre=r.get("strategy_preflight",{}).get("visual_then_balanced",{})
    if r.get("language")==lang and r.get("task_domain")==domain and r.get("question_type")=="open" and sid not in used and r.get("query_image") and Path(r["query_image"]).is_file() and pre.get("eligible") and int(pre.get("rank",99))==1 and float(pre.get("score",0))>=.70: opts.append(r)
  if not opts: raise SystemExit(f"no target for {lang}/{domain}")
  r=dict(sorted(opts,key=lambda x:(-float(x.get("strategy_preflight",{}).get("visual_then_balanced",{}).get("score",0)),str(x.get("source_sample_id"))))[0])
  r.update({"sample_id":r.get("source_sample_id"),"target_id":f"rag_correction-round036-{lang}-{domain}-{r.get('source_sample_id')}","trajectory_mode":"stop_correction","desired_correction_type":desired,"generation_route":route,"label_visible_to_teacher":route=="oracle_grounded","strategy_id":"visual_then_balanced","preferred_sequence":["visual","balanced"],"round":36,"focus":["typed_correction",desired,"rank1_margin","language_isolation"]})
  chosen.append(r)
out=root/"rounds/round_0001/plan/round036_correction_mix_plan.jsonl"
out.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in chosen),encoding="utf-8")
print(json.dumps({"rows":len(chosen),"ids":[r.get("source_sample_id") for r in chosen],"types":[r["desired_correction_type"] for r in chosen]},ensure_ascii=False))
