#!/usr/bin/env python3
"""Offline repair of a completed private-final pilot after a gate fix."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
from agrinet.research.hcv.collector import (
    ACCEPTANCE_POLICY, accept_trajectory, canonical_label_name, class_name_aliases,
    correction_audit, generation_route, label_influence_audit, sample_strategy,
    unique_reference_images, write_jsonl,
)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-audit", type=Path, required=True)
    return parser.parse_args()

def row_from_trace(trace: dict[str, Any]) -> dict[str, Any]:
    sample = trace["sample"]
    messages = trace["sft_messages"]
    retrievals = trace.get("retrieval_calls") or []
    accepted, reasons = accept_trajectory(sample, messages, retrievals)
    if not accepted:
        raise RuntimeError(f"trace remains rejected: {sample.get('sample_id')}: {reasons}")
    private = trace.get("private_final_adjudication")
    if not isinstance(private, dict) or private.get("mode") != "private_final_adjudication" or private.get("public_evidence_grounded") is not True:
        raise RuntimeError(f"trace lacks public-grounded private final metadata: {sample.get('sample_id')}")
    retrieval_types = [item.get("tool_call", {}).get("arguments", {}).get("retrieval_type") for item in retrievals]
    metadata = {
        "label_code": sample.get("final_label"),
        "canonical_class": sample.get("canonical_class") or sample.get("final_label"),
        "label_name": canonical_label_name(sample),
        "label_name_zh": sample.get("final_label_zh"),
        "label_aliases": class_name_aliases(sample),
        "task_domain": sample.get("task_domain"),
        "accepted": True,
        "acceptance_policy": ACCEPTANCE_POLICY,
        "retrieval_turns": len(retrievals),
        "retrieval_types": retrieval_types,
        "strategy_id": sample.get("strategy_id", "legacy_visual"),
        "preferred_sequence": list(sample_strategy(sample, 3)[1]),
        "retrieval_top_k": 3,
        "generation_route": generation_route(sample),
        "label_visible_to_teacher": False,
        "label_influence_audit": label_influence_audit(messages, class_name_aliases(sample)),
        "correction_audit": correction_audit(sample, messages),
        "language": sample.get("language", "en"),
        "question_type": sample.get("question_type", "open"),
        "trajectory_mode": sample.get("trajectory_mode", "standard"),
        "correct_option": sample.get("correct_option"),
        "candidate_labels": sample.get("public_option_labels") or sample.get("candidate_labels"),
        "image_sha256": sample.get("image_sha256"),
        "private_final_adjudication": private,
    }
    return {"sample_id": sample.get("sample_id"), "tools": sample.get("tools") or [], "messages": messages, "images": [sample["query_image"], *unique_reference_images(retrievals)], "metadata": metadata}

def main() -> int:
    args = parse_args()
    source = args.source_dir
    if args.output_dir.exists():
        raise SystemExit(f"output already exists: {args.output_dir}")
    traces = read_jsonl(source / "traces/raw_trajectories.jsonl")
    private_ids = {str(row.get("sample_id") or "") for row in read_jsonl(args.private_audit)}
    repaired = []
    for trace in traces:
        if trace.get("accepted"):
            continue
        row = row_from_trace(trace)
        if str(row.get("sample_id") or "") not in private_ids:
            raise SystemExit(f"missing private audit row: {row.get('sample_id')}")
        repaired.append(row)
    original = read_jsonl(source / "train/agent_sft.accepted.jsonl")
    all_rows = original + repaired
    ids = [str(row.get("sample_id") or "") for row in all_rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate sample IDs after repair")
    (args.output_dir / "train").mkdir(parents=True)
    (args.output_dir / "reports").mkdir()
    write_jsonl(args.output_dir / "train/agent_sft.accepted.jsonl", all_rows)
    write_jsonl(args.output_dir / "train/agent_sft.blind_evidence.jsonl", all_rows)
    report = {
        "schema_version": "agrinet.private-final-pilot-repair/v1",
        "source_dir": str(source), "original_accepted": len(original),
        "repaired_rows": len(repaired), "total_accepted": len(all_rows),
        "teacher_recontacted": False, "retrieval_reexecuted": False,
    }
    (args.output_dir / "reports/repair_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"repaired": len(repaired), "total_accepted": len(all_rows), "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
