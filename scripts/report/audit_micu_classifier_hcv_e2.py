#!/usr/bin/env python3
"""Produce the independent E2 quality, cost, and learnability audit."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tool_counts(trajectory: dict[str, Any]) -> Counter[str]:
    result: Counter[str] = Counter()
    for event in trajectory.get("trace") or []:
        tool = event.get("tool") if isinstance(event, dict) else None
        if isinstance(tool, dict) and tool.get("tool"):
            result[str(tool["tool"])] += 1
    return result


def trajectory_diagnostics(source: dict[str, Any], trajectory: dict[str, Any]) -> dict[str, Any]:
    truth = str(source["private"]["truth_code"])
    rag_calls = 0; rag_truth_hits = 0
    for event in trajectory.get("trace") or []:
        tool = event.get("tool") if isinstance(event, dict) else None
        if not isinstance(tool, dict) or tool.get("tool") != "agrinet_rag_search":
            continue
        rag_calls += 1
        evidence = (tool.get("raw_response") or {}).get("evidence") or []
        codes = {str((item.get("metadata") or {}).get("code") or "") for item in evidence}
        rag_truth_hits += truth in codes
    normalized = trajectory.get("normalized_final") or {}
    if normalized.get("answer_status") == "insufficient_evidence":
        stop = "insufficient_evidence"
    elif normalized.get("answer_status") == "answered":
        stop = "answered"
    else:
        stop = "invalid_or_missing_answer"
    return {"rag_calls": rag_calls, "rag_truth_hits": rag_truth_hits, "stop": stop,
            "tool_calls": sum(tool_counts(trajectory).values()),
            "trace_events": len(trajectory.get("trace") or [])}


def grouped_parent_metrics(
    by_sample: dict[str, dict[str, Any]], trajectories: dict[str, dict[str, Any]], audits: dict[str, dict[str, Any]],
    key_fn: Any,
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for sample_id, source in by_sample.items():
        key = str(key_fn(source))
        bucket = buckets.setdefault(key, {"source_images": 0, "closed_parents": 0, "tool_calls": 0,
                                          "rag_calls": 0, "audit_statuses": Counter()})
        bucket["source_images"] += 1
        trajectory = trajectories.get(sample_id)
        if trajectory is not None:
            diagnostics = trajectory_diagnostics(source, trajectory)
            bucket["closed_parents"] += 1
            bucket["tool_calls"] += diagnostics["tool_calls"]
            bucket["rag_calls"] += diagnostics["rag_calls"]
        audit = audits.get(sample_id)
        if audit is not None:
            bucket["audit_statuses"][str(audit.get("status"))] += 1
    result = {}
    for key, bucket in sorted(buckets.items()):
        closed = bucket["closed_parents"]
        audited = sum(bucket["audit_statuses"].values())
        result[key] = {
            "source_images": bucket["source_images"], "closed_parents": closed,
            "closed_fraction": closed / bucket["source_images"] if bucket["source_images"] else None,
            "mean_tool_calls_per_closed": bucket["tool_calls"] / closed if closed else None,
            "mean_rag_calls_per_closed": bucket["rag_calls"] / closed if closed else None,
            "audit_statuses": dict(sorted(bucket["audit_statuses"].items())),
            "accept_fraction_of_audited": bucket["audit_statuses"].get("accept", 0) / audited if audited else None,
        }
    return result


def usage_from_ledgers(root: Path) -> dict[str, Any]:
    usage: Counter[str] = Counter(); statuses: Counter[str] = Counter(); latencies: list[float] = []
    for path in root.glob("**/ledger/events.jsonl"):
        for event in read_jsonl(path):
            if event.get("event") != "result":
                continue
            statuses[str(event.get("status") or "unknown")] += 1
            token_usage = event.get("usage") or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if isinstance(token_usage.get(key), (int, float)):
                    usage[key] += int(token_usage[key])
            elapsed = event.get("elapsed_seconds") or event.get("latency_seconds")
            if isinstance(elapsed, (int, float)):
                latencies.append(float(elapsed))
    return {"tokens": dict(usage), "request_statuses": dict(statuses), "latency_seconds": {
        "count": len(latencies), "mean": statistics.fmean(latencies) if latencies else None,
        "max": max(latencies) if latencies else None}}


def retry_candidates_from_ledgers(root: Path, quarantined: set[str]) -> list[dict[str, Any]]:
    """Record ambiguous upstream delivery as retry work, never as data-quality failure."""
    candidates = []
    patterns = (
        ("parent", "parent/public/*/ledger/events.jsonl"),
        ("parent_audit", "private/*/parent/ledger/events.jsonl"),
        ("g1", "derivations/g1/*/ledger/events.jsonl"),
        ("g2", "derivations/g2/*/ledger/events.jsonl"),
        ("g1_audit", "private/*/g1/ledger/events.jsonl"),
        ("g2_audit", "private/*/g2/ledger/events.jsonl"),
    )
    for scope, pattern in patterns:
        for path in root.glob(pattern):
            sample_id = path.parent.parent.name if scope in {"parent", "g1", "g2"} else path.parents[2].name
            events = read_jsonl(path)
            intents = {str(row["request_key"]): row for row in events if row.get("event") == "intent"}
            results = {str(row["request_key"]): row for row in events if row.get("event") == "result"}
            for request_key, intent in intents.items():
                result = results.get(request_key)
                status = str(result.get("status")) if result is not None else "unresolved_intent"
                if status not in {"unknown_delivery", "invalid_response", "truncated", "unresolved_intent"}:
                    continue
                candidates.append({
                    "sample_id": sample_id, "scope": scope, "request_key": request_key,
                    "original_request_id": intent.get("request_id"), "original_status": status,
                    "error_type": result.get("error_type") if result else None,
                    "latency_seconds": result.get("latency_seconds") if result else None,
                    "retry_state": "quarantined_no_retry" if sample_id in quarantined else "retry_pending",
                    "automatic_replay_allowed": False,
                    "retry_requires_new_attempt_id": True,
                })
    return sorted(candidates, key=lambda row: (row["scope"], row["sample_id"], row["request_key"]))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    source = read_jsonl(args.source)
    by_sample = {str(row["sample_id"]): row for row in source}
    trajectories = {}
    for path in args.artifact_root.glob("parent/public/*/trajectory.json"):
        row = read_json(path); trajectories[str(row["sample_id"])] = row
    leaked_p6_samples = []
    for path in args.artifact_root.glob("parent/public/*/ledger/events.jsonl"):
        if "excluded_supervised_codes" in path.read_text(encoding="utf-8"):
            leaked_p6_samples.append(path.parent.parent.name)
    audits = {}
    for path in args.artifact_root.glob("private/*/parent/audit.json"):
        row = read_json(path); audits[str(row["sample_id"])] = row
    derivation = read_json(args.artifact_root / "derivations/summary.json") if (args.artifact_root / "derivations/summary.json").is_file() else {}
    parent_batch = read_json(args.artifact_root / "parent/summary.json") if (args.artifact_root / "parent/summary.json").is_file() else {}
    boundary_path = args.artifact_root / "public_boundary_exclusions.json"
    boundary = read_json(boundary_path) if boundary_path.is_file() else {"sample_ids": []}
    boundary_ids = set(boundary.get("sample_ids") or [])
    retry_candidates = retry_candidates_from_ledgers(args.artifact_root, boundary_ids)

    arms = Counter(str(row["private"].get("sampling_arm")) for row in source)
    target_patterns = Counter(str(row["private"].get("target_pattern")) for row in source)
    observed_patterns = Counter(str(row.get("primary_pattern")) for row in audits.values())
    observed_by_prediction = Counter()
    for sample_id, audit in audits.items():
        source_row = by_sample.get(sample_id)
        if source_row is not None:
            kind = str(source_row.get("prediction", {}).get("kind"))
            observed_by_prediction[f"{kind}:{audit.get('primary_pattern')}"] += 1
    cells = Counter("-".join(str(row[k]) for k in ("question_type", "language", "task_domain")) for row in source)
    prediction_kinds = Counter(str(row.get("prediction", {}).get("kind")) for row in source)
    p6_rows = [row for row in source if row.get("prediction", {}).get("kind") == "p6_class_holdout"]
    p6_groups = Counter(str(row["prediction"].get("p6_group")) for row in p6_rows)
    p6_classes = Counter(str(row["private"].get("truth_code")) for row in p6_rows)
    parent_statuses = Counter(str(row.get("status")) for row in trajectories.values())
    batch_image_statuses = Counter(str(row.get("status")) for row in parent_batch.get("statuses") or []
                                   if row.get("sample_id") is not None)
    audit_statuses = Counter(str(row.get("status")) for row in audits.values())
    tools: Counter[str] = Counter()
    stops: Counter[str] = Counter(); arm_audits: Counter[str] = Counter()
    rag_calls = 0; rag_truth_hits = 0; tool_lengths: list[int] = []
    for trajectory in trajectories.values():
        tools.update(tool_counts(trajectory))
        source_row = by_sample.get(str(trajectory.get("sample_id")))
        if source_row is not None:
            diagnostics = trajectory_diagnostics(source_row, trajectory)
            stops[diagnostics["stop"]] += 1
            rag_calls += diagnostics["rag_calls"]; rag_truth_hits += diagnostics["rag_truth_hits"]
            tool_lengths.append(diagnostics["tool_calls"])
    for sample_id, audit in audits.items():
        source_row = by_sample.get(sample_id)
        if source_row is not None:
            arm = str(source_row["private"].get("sampling_arm"))
            arm_audits[f"{arm}:{audit.get('status')}"] += 1

    option_errors = 0
    for sample_id, trajectory in trajectories.items():
        source_row = by_sample.get(sample_id)
        if source_row and source_row.get("question_type") == "option":
            normalized = trajectory.get("normalized_final") or {}
            if normalized.get("answer_status") == "answered" and not normalized.get("selected_option"):
                option_errors += 1
    selected = derivation.get("selected") or []
    parent_audit_rows = [row for row in derivation.get("statuses") or [] if row.get("stage") == "parent_audit"]
    derivation_statuses = [row for row in derivation.get("statuses") or [] if row.get("stage") in {"g1", "g2"}]
    batch_status_by_id = {str(row["sample_id"]): str(row.get("status"))
                          for row in parent_batch.get("statuses") or [] if row.get("sample_id") is not None}
    parent_audit_by_id = {str(row["sample_id"]): row for row in parent_audit_rows if row.get("sample_id") is not None}
    selected_ids = {str(row.get("sample_id")) for row in selected}
    derivation_accounting = {(str(row.get("sample_id")), str(row.get("stage"))) for row in derivation_statuses}
    expected_audits = len(set(trajectories) - boundary_ids)
    final_lengths = [len(str(row.get("final") or "")) for row in trajectories.values()]
    trace_lengths = [len(json.dumps(row.get("trace") or [], ensure_ascii=False))
                     for row in trajectories.values()]
    report = {
        "schema_version": "agrinet.micu-classifier-hcv-e2-audit/v1",
        "training_eligible": False,
        "public_boundary_exclusions": boundary,
        "public_boundary_audit": {"leaked_p6_samples": sorted(leaked_p6_samples),
                                  "all_leaks_quarantined": set(leaked_p6_samples) <= boundary_ids},
        "retry": {
            "policy": "upstream delivery failures are retry work, not content-quality failures",
            "automatic_replay_allowed": False,
            "requires_separate_attempt_with_new_request_id": True,
            "counts": dict(sorted(Counter(row["retry_state"] for row in retry_candidates).items())),
            "by_scope": dict(sorted(Counter(row["scope"] for row in retry_candidates).items())),
            "by_original_status": dict(sorted(Counter(row["original_status"] for row in retry_candidates).items())),
        },
        "source": {"rows": len(source), "unique_samples": len(by_sample),
                   "cells": dict(sorted(cells.items())), "arms": dict(sorted(arms.items())),
                   "target_patterns": dict(sorted(target_patterns.items())),
                   "prediction_kinds": dict(sorted(prediction_kinds.items())),
                   "p6_simulated_unknown": {"rows": len(p6_rows),
                       "groups": dict(sorted(p6_groups.items())),
                       "covered_holdout_classes": len(p6_classes), "available_holdout_classes": 24,
                       "class_counts": dict(sorted(p6_classes.items())),
                       "coverage_scope": "process exploration only; not formal Unknown generalization"}},
        "parents": {"rows": len(trajectories), "statuses": dict(sorted(parent_statuses.items())),
                    "attempted_image_statuses": dict(sorted(batch_image_statuses.items())),
                    "attempted_images": sum(batch_image_statuses.values()),
                    "tool_counts": dict(sorted(tools.items())), "stop_reasons": dict(sorted(stops.items())),
                    "mean_tool_calls": statistics.fmean(tool_lengths) if tool_lengths else None,
                    "rag_truth_hit_fraction": rag_truth_hits / rag_calls if rag_calls else None,
                    "option_mapping_errors": option_errors},
        "comparisons": {
            "sampling_arm": grouped_parent_metrics(
                by_sample, trajectories, audits, lambda row: row["private"].get("sampling_arm")),
            "prediction_kind": grouped_parent_metrics(
                by_sample, trajectories, audits, lambda row: row.get("prediction", {}).get("kind")),
            "target_pattern": grouped_parent_metrics(
                by_sample, trajectories, audits, lambda row: row["private"].get("target_pattern")),
        },
        "private_audits": {"rows": len(audits), "statuses": dict(sorted(audit_statuses.items())),
                           "arm_decisions": dict(sorted(arm_audits.items())),
                           "observed_primary_patterns": dict(sorted(observed_patterns.items())),
                           "observed_pattern_by_prediction_kind": dict(sorted(observed_by_prediction.items()))},
        "derivations": {"selected_images": len(selected), "selected": selected,
                        "statuses": derivation.get("statuses") or [],
                        "selection_exclusions": derivation.get("selection_exclusions") or []},
        "cost": {"reserved_micu_requests": len(read_jsonl(args.artifact_root / "global_micu_events.jsonl")),
                 "reserved_rag_calls": len(read_jsonl(args.artifact_root / "global_rag_events.jsonl")),
                 "ledger_usage": usage_from_ledgers(args.artifact_root)},
    }
    qwen_rows = [row for path in args.qwen for row in read_jsonl(path)]
    qwen_seconds = sum(float(row.get("elapsed_seconds") or 0) for row in qwen_rows)
    report["prescreen"] = {
        "qwen_rows": len(qwen_rows),
        "qwen_complete": sum(row.get("status") == "complete" for row in qwen_rows),
        "qwen_aggregate_images_per_second": len(qwen_rows) / qwen_seconds if qwen_seconds else None,
        "rag": read_json(args.rag_summary) if args.rag_summary.is_file() else "not_recorded",
    }
    p6_groups = []
    for group, root in enumerate(args.p6_root):
        history_path = root / "classifier/classifier_dev_metrics.json"
        predictions_path = root / "classifier/predictions_dev_known.jsonl"
        history = read_json(history_path) if history_path.is_file() else []
        best = max(history, key=lambda row: row.get("macro_f1", -1)) if history else {}
        p6_groups.append({"group": group, "epochs": len(history),
                          "best_epoch": best.get("epoch"), "best_macro_f1": best.get("macro_f1"),
                          "holdout_predictions": len(read_jsonl(predictions_path))})
    report["p6_classifiers"] = p6_groups
    g1_rows = len(list(args.artifact_root.glob("derivations/g1/*/derivation.json")))
    g2_rows = len(list(args.artifact_root.glob("derivations/g2/*/derivation.json")))
    report["derivations"].update({"g1_rows": g1_rows, "g2_rows": g2_rows})
    report["learnability"] = {
        "student_model": "models/Qwen3-VL-4B-Instruct original weights",
        "teacher_rewrite_used_for_g1": True,
        "one_observed_pattern_per_query": len(audits) == sum(observed_patterns.values()),
        "natural_teacher_samples_only": True,
        "teacher_token_logits_available": False,
        "approximate_real_distribution_basis": "sampled deterministic teacher trajectories, not handwritten reasoning",
        "parent_final_length_chars": {
            "count": len(final_lengths),
            "mean": statistics.fmean(final_lengths) if final_lengths else None,
            "min": min(final_lengths) if final_lengths else None,
            "max": max(final_lengths) if final_lengths else None,
        },
        "parent_trace_serialized_chars": {
            "mean": statistics.fmean(trace_lengths) if trace_lengths else None,
            "min": min(trace_lengths) if trace_lengths else None,
            "max": max(trace_lengths) if trace_lengths else None,
        },
        "sft_started": False,
        "training_eligible": False,
    }
    report["completion_ready"] = (
        len(source) == 160 and arms == {"targeted": 80, "random": 80}
        and set(batch_status_by_id) == set(by_sample) and len(batch_status_by_id) == 160
        and set(parent_audit_by_id) == set(by_sample) and len(parent_audit_by_id) == 160
        and len(audits) <= expected_audits and sum(observed_patterns.values()) == len(audits)
        and set(leaked_p6_samples) <= boundary_ids
        and len(selected_ids) == len(selected) <= 20
        and all(sum(str(row.get("primary_pattern")) == f"P{pattern}" for row in selected) <= 2
                for pattern in range(1, 11))
        and derivation_accounting == {(sample_id, stage) for sample_id in selected_ids for stage in ("g1", "g2")}
        and all(item["epochs"] == 50 and item["holdout_predictions"] > 0 for item in p6_groups)
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, action="append", default=[])
    parser.add_argument("--rag-summary", type=Path, required=True)
    parser.add_argument("--p6-root", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--output-retry", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.p6_root) != 3:
        raise ValueError("exactly three P6 artifact roots are required")
    report = build_report(args)
    boundary = set(report.get("public_boundary_exclusions", {}).get("sample_ids") or [])
    retry_candidates = retry_candidates_from_ledgers(args.artifact_root, boundary)
    retry_path = args.output_retry or args.output_json.with_name("retry_pending.jsonl")
    retry_path.parent.mkdir(parents=True, exist_ok=True)
    retry_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                    for row in retry_candidates), encoding="utf-8")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Micu × classifier × HCV E2 audit",
        "",
        f"- Completion ready: `{report['completion_ready']}`",
        f"- Training eligible: `{report['training_eligible']}`",
        f"- Source arms: `{report['source']['arms']}`",
        f"- Parent trajectories: `{report['parents']['rows']}/160`",
        f"- Parent audits: `{report['private_audits']['rows']}/160`",
        f"- G1/G2: `{report['derivations']['g1_rows']}/{report['derivations']['g2_rows']}`",
        f"- Micu/RAG reservations: `{report['cost']['reserved_micu_requests']}/{report['cost']['reserved_rag_calls']}`",
        f"- Retry pending/quarantined: `{report['retry']['counts']}`",
        "",
        "## Observed patterns",
        "",
    ]
    for key, value in report["private_audits"]["observed_primary_patterns"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## P6 classifiers", ""])
    for item in report["p6_classifiers"]:
        lines.append(
            f"- Group {item['group']}: best epoch {item['best_epoch']}, "
            f"Macro-F1 {item['best_macro_f1']}, holdout rows {item['holdout_predictions']}"
        )
    lines.extend(["", "## Targeted versus random", ""])
    for arm, item in report["comparisons"]["sampling_arm"].items():
        lines.append(
            f"- {arm}: closed {item['closed_parents']}/{item['source_images']}, "
            f"audit accept fraction {item['accept_fraction_of_audited']}, "
            f"mean tools/RAG per closed {item['mean_tool_calls_per_closed']}/{item['mean_rag_calls_per_closed']}"
        )
    lines.extend(["", "## Learnability limitations", "",
                  "- G1 reasoning is model-generated from the full public trace; it is not handwritten.",
                  "- Teacher token logits are unavailable, so this run supports sampled approximate-distribution analysis only, not exact distribution matching.",
                  "- No SFT was started and every artifact remains training-ineligible."])
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output_json),
                      "completion_ready": report["completion_ready"]}, ensure_ascii=False))
    return 0 if report["completion_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
