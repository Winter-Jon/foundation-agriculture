#!/usr/bin/env python3
"""Audit label-blind retrieval coverage in a completed formal RAG run."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _load(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row.get("id") or "")
        if not sample_id or sample_id in rows:
            raise SystemExit(f"{path}: empty or duplicate id")
        rows[sample_id] = row
    return rows


def _evidence(row: dict[str, Any]) -> dict[str, Any]:
    aliases = {_norm(value) for value in row.get("label_aliases", []) if _norm(value)}
    if not aliases:
        aliases.add(_norm(row.get("label_name", "")))
    ranks: list[int] = []
    total = 0
    for event in row.get("tool_history", []):
        response = event.get("tool_response", {})
        for hit in response.get("results", []) or []:
            if not isinstance(hit, dict):
                continue
            total += 1
            terms = [hit.get(key, "") for key in ("class_name", "chinese_name", "code")]
            terms.extend(hit.get("aliases", []) or [])
            terms.extend(hit.get("chinese_aliases", []) or [])
            if any(_norm(term) in aliases for term in terms if _norm(term)):
                ranks.append(int(hit.get("rank") or 0))
    return {
        "truth_hit": bool(ranks),
        "best_within_call_rank": min((rank for rank in ranks if rank > 0), default=None),
        "returned_hits": total,
    }


def _summary(rows: list[dict[str, Any]], baseline: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    evidence = [_evidence(row) for row in rows]
    hit_rows = [item for item in evidence if item["truth_hit"]]
    hit_correct = sum(bool(row.get("correct")) for row, item in zip(rows, evidence) if item["truth_hit"])
    miss_correct = sum(bool(row.get("correct")) for row, item in zip(rows, evidence) if not item["truth_hit"])
    result = {
        "rows": len(rows),
        "rag_correct": sum(bool(row.get("correct")) for row in rows),
        "truth_hit_rows": len(hit_rows),
        "truth_miss_rows": len(rows) - len(hit_rows),
        "truth_hit_rate": len(hit_rows) / len(rows) if rows else 0.0,
        "rag_correct_given_truth_hit": hit_correct / len(hit_rows) if hit_rows else None,
        "rag_correct_given_truth_miss": miss_correct / (len(rows) - len(hit_rows)) if len(rows) > len(hit_rows) else None,
        "mean_returned_hits": sum(item["returned_hits"] for item in evidence) / len(rows) if rows else 0.0,
        "mean_best_within_call_rank_when_hit": (
            sum(item["best_within_call_rank"] for item in hit_rows) / len(hit_rows)
            if hit_rows else None
        ),
        "invalid_tool_call_rows": sum(bool(row.get("protocol", {}).get("has_invalid_tool_call")) for row in rows),
        "forced_fallback_rows": sum(int(row.get("forced_tool_turns", 0)) > 0 for row in rows),
    }
    if baseline is not None:
        result.update({
            "baseline_correct": sum(bool(baseline[row["id"]].get("correct")) for row in rows),
            "rag_only_correct": sum(bool(row.get("correct")) and not bool(baseline[row["id"]].get("correct")) for row in rows),
            "baseline_only_correct": sum(not bool(row.get("correct")) and bool(baseline[row["id"]].get("correct")) for row in rows),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-scored", type=Path, required=True)
    parser.add_argument("--baseline-scored", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rag = _load(args.rag_scored)
    baseline = _load(args.baseline_scored) if args.baseline_scored else None
    if baseline is not None and set(rag) != set(baseline):
        raise SystemExit("RAG and baseline ids differ")

    groups: dict[str, list[dict[str, Any]]] = {"overall": list(rag.values())}
    for domain in ("disease", "pest"):
        for question_type in ("open", "option"):
            key = f"{domain}/{question_type}"
            groups[key] = [
                row for row in rag.values()
                if row.get("task_domain") == domain and row.get("question_type") == question_type
            ]
    if baseline is not None:
        groups["pest/option/baseline_only"] = [
            row for row in groups["pest/option"]
            if not bool(row.get("correct")) and bool(baseline[row["id"]].get("correct"))
        ]
        groups["pest/option/rag_only"] = [
            row for row in groups["pest/option"]
            if bool(row.get("correct")) and not bool(baseline[row["id"]].get("correct"))
        ]
    result = {
        "schema_version": "agrinet.formal-rag-trajectory-audit/v1",
        "rag_scored": str(args.rag_scored),
        "baseline_scored": str(args.baseline_scored) if args.baseline_scored else None,
        "groups": {key: _summary(rows, baseline) for key, rows in groups.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value["rows"] for key, value in result["groups"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
