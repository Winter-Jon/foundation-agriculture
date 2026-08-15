#!/usr/bin/env python3
"""Create an auditable baseline and first bounded iteration plan."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--nonrag", type=Path, required=True)
    parser.add_argument("--prior-rag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.baseline; output = args.output
    predictions = read_jsonl(base / "predictions.jsonl")
    report = read_json(base / "final_report.json")
    nonrag = read_json(args.nonrag / "final_report.json")
    prior = read_json(args.prior_rag / "final_report.json")
    calls = successful = forced_rows = spontaneous_rows = 0
    retrieval_types = Counter()
    for row in predictions:
        turns = int(row.get("tool_turns", 0)); forced = int(row.get("forced_tool_turns", 0))
        forced_rows += forced > 0; spontaneous_rows += turns > forced
        for history in row.get("tool_history", []):
            calls += 1; successful += bool(history.get("ledger", {}).get("ok"))
            retrieval_types[history.get("tool_call", {}).get("arguments", {}).get("retrieval_type")] += 1
    baseline = {
        "records": len(predictions),
        "accuracy": report["overall"]["accuracy"],
        "nonrag_accuracy": nonrag["overall"]["accuracy"],
        "prior_rag_accuracy": prior["overall"]["accuracy"],
        "gap_to_nonrag_pp": 100 * (report["overall"]["accuracy"] - nonrag["overall"]["accuracy"]),
        "tool_calls": calls, "successful_tool_calls": successful,
        "forced_rows": forced_rows, "spontaneous_rows": spontaneous_rows,
        "spontaneous_rate": spontaneous_rows / len(predictions),
        "retrieval_types": dict(retrieval_types),
        "report": report,
        "sha256": {"predictions": sha256(base / "predictions.jsonl"), "report": sha256(base / "final_report.json")},
    }
    write_json(output / "baseline/metrics.json", baseline)
    write_json(output / "baseline/failure_buckets.json", [
        {"id": "autonomous_first_tool_call", "priority": "P0", "observed_rate": baseline["spontaneous_rate"], "target_rate": 0.80},
        {"id": "option_answer_retention", "priority": "P0", "observed_accuracy": report["known/option"]["accuracy"], "nonrag_accuracy": nonrag["known/option"]["accuracy"]},
    ])
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    state = {"schema_version": "agrinet.iteration-state/v1", "status": "awaiting_round_1_pilot", "round": 1, "stage": "plan_delta", "updated_at": timestamp, "training_authorized": False, "full_eval_authorized": False, "next_action": "materialize and validate a 16-attempt autonomous-routing and Option pilot"}
    write_json(output / "state.json", state)
    write_json(output / "active_round.json", {"round": 1, "focus": ["autonomous_first_tool_call", "option_answer_retention"], "attempts": 16, "status": "planned", "training_authorized": False})
    write_json(output / "rounds/round_0001/plan/round_plan.json", {"round": 1, "hypothesis": "Explicit first-call and evidence-to-option supervision improves routing and Option retention without relaxing hard gates.", "attempts": 16, "training_authorized": False})
    review = f"# Round 0 Baseline Review\n\nStatus: verified\n\n- Corrected RAG: {report['overall']['correct']}/{len(predictions)} = {report['overall']['accuracy']:.2%}\n- Non-RAG checkpoint-165: {nonrag['overall']['accuracy']:.2%}; gap {baseline['gap_to_nonrag_pp']:+.2f} pp\n- Prior RAG: {prior['overall']['accuracy']:.2%}\n- Milvus calls: {successful}/{calls} successful\n- Spontaneous-call samples: {spontaneous_rows}/{len(predictions)} = {baseline['spontaneous_rate']:.2%}\n- First-call fallback samples: {forced_rows}/{len(predictions)}\n\n## Next\n\nRound 1 is limited to 16 attempts and targets autonomous first tool calls plus Option answer retention. No SFT is authorized until all hard gates pass and at least 16 novel accepted rows exist.\n"""
    (output / "baseline/review.md").parent.mkdir(parents=True, exist_ok=True)
    (output / "baseline/review.md").write_text(review, encoding="utf-8")
    with (output / "loop_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": timestamp, "event": "iteration_initialized", "state": state}, ensure_ascii=False) + "\n")
    print(json.dumps({"status": state["status"], "records": len(predictions), "output": str(output)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
