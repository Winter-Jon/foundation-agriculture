from __future__ import annotations
import glob, json
from pathlib import Path

TARGETS = {
    "pest_N05013_N05013_P00005",
    "pest_N05014_N05014_P00004",
    "pest_N05010_N05010_P00003",
    "disease_N04070_N04070_P00006",
}
CONFIG = [
    ("pest_N05013_N05013_P00005", "visual_then_balanced", "oracle_grounded"),
    ("pest_N05014_N05014_P00004", "semantic_then_visual", "oracle_grounded"),
    ("pest_N05010_N05010_P00003", "visual_then_balanced", "oracle_grounded"),
    ("disease_N04070_N04070_P00006", "visual_then_balanced", "blind_evidence"),
]
SEQUENCES = {"visual_then_balanced": ["visual", "balanced"], "semantic_then_visual": ["semantic", "visual"]}

def main() -> None:
    found: dict[str, dict] = {}
    pattern = "outputs/artifacts/datasets/agrinet-rag-recovery-pilot-v5/**/traces/raw_trajectories.jsonl"
    for path in glob.glob(pattern, recursive=True):
        for line in Path(path).open(encoding="utf-8"):
            row = json.loads(line)
            sample = row.get("sample") or {}
            sid = sample.get("source_sample_id")
            if sid in TARGETS and sample.get("candidate_labels") and sid not in found:
                found[sid] = sample
    missing = TARGETS - found.keys()
    if missing:
        raise SystemExit(f"missing option source rows: {sorted(missing)}")
    base = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan")
    source = base / "round007_option_source.jsonl"
    source_rows = [{**found[sid], "sample_id": sid} for sid in found]
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows), encoding="utf-8")
    plans = []
    for sid, strategy, route in CONFIG:
        sample = found[sid]
        plans.append({**sample, "target_id": f"rag_option-round007-{sid}", "source_sample_id": sid,
            "sample_id": f"rag_option-round007-{sid}", "generation_route": route,
            "strategy_id": strategy, "preferred_sequence": SEQUENCES[strategy], "round": 7,
            "preflight_eligible": True, "strict_first_tool_call": True})
    plan = base / "round007_option_plan.jsonl"
    plan.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in plans), encoding="utf-8")
    print(json.dumps({"source": str(source), "plan": str(plan), "rows": len(plans),
        "coverage": [{"id": r["target_id"], "language": r["language"], "domain": r["task_domain"],
        "option": r["correct_option"], "strategy": r["strategy_id"], "route": r["generation_route"]} for r in plans]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
