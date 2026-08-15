"""Build a tiny contract-valid pilot: two complete Options and one Chinese Open."""
from pathlib import Path
import json

option_src = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round007_option_source.jsonl")
open_src = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round014_zh_pest_open_source.jsonl")
rows = [json.loads(x) for x in option_src.read_text(encoding="utf-8").splitlines() if x.strip()]
rows += [json.loads(x) for x in open_src.read_text(encoding="utf-8").splitlines() if x.strip()]
source = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round078_contract_source.jsonl")
source.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")
selected = []
for r in rows:
    if r.get("question_type") == "option" and r.get("language") == "en" and r.get("sample_id") == "pest_N05014_N05014_P00004":
        x = dict(r); x["target_id"] = "rag_option-round078-en-pest_N05014"; x["generation_route"] = "blind_evidence"; x["label_visible_to_teacher"] = False; selected.append(x)
    elif r.get("question_type") == "option" and r.get("language") == "zh" and r.get("sample_id") == "pest_N05010_N05010_P00003":
        x = dict(r); x["target_id"] = "rag_option-round078-zh-pest_N05010"; x["generation_route"] = "blind_evidence"; x["label_visible_to_teacher"] = False; selected.append(x)
    elif r.get("question_type") == "open":
        x = dict(r); x["target_id"] = "rag_open-round078-zh-pest_N05070"; x["trajectory_mode"] = "standard"; x["strategy_id"] = "semantic_then_visual"; x["preferred_sequence"] = ["semantic", "visual"]; x["generation_route"] = "blind_evidence"; x["label_visible_to_teacher"] = False; x["desired_correction_type"] = None; selected.append(x)
plan = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round078_contract_repair_plan.jsonl")
plan.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in selected), encoding="utf-8")
print(json.dumps({"source": str(source), "plan": str(plan), "selected": [{k:r.get(k) for k in ("sample_id", "language", "task_domain", "question_type", "strategy_id", "generation_route")} for r in selected]}, ensure_ascii=False))
