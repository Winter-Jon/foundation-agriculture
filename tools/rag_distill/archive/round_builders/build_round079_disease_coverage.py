"""Select two unused Disease Open rows with preflight-aligned strategies."""
from pathlib import Path
import json

src = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/supplement_plan_strict_next.jsonl")
rows = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
wanted = {"disease_N04011_N04011_P00005", "disease_N04059_N04059_P00001"}
selected = []
for r in rows:
    if r.get("source_sample_id") in wanted:
        x = dict(r)
        x["sample_id"] = r["source_sample_id"]
        x["target_id"] = "rag_open-round079-" + r["source_sample_id"]
        x["trajectory_mode"] = "standard"
        x["generation_route"] = "blind_evidence"
        x["label_visible_to_teacher"] = False
        selected.append(x)
if len(selected) != len(wanted):
    raise SystemExit(f"expected {len(wanted)} rows, found {len(selected)}")
source = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round079_disease_source.jsonl")
plan = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round079_disease_plan.jsonl")
payload = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in selected)
source.write_text(payload, encoding="utf-8")
plan.write_text(payload, encoding="utf-8")
print(json.dumps({"source": str(source), "plan": str(plan), "selected": [{k:r.get(k) for k in ("source_sample_id", "language", "task_domain", "question_type", "strategy_id", "preferred_sequence")} for r in selected]}, ensure_ascii=False))
