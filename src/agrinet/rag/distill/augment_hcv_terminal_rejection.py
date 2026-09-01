#!/usr/bin/env python3
"""Add label-blind rejected-tool terminal states to frozen HCV SFT rows.

The appended trajectories never execute a synthetic retrieval. They reproduce
the evaluator's public ``tool_budget_exhausted`` observation and reuse only the
source row's already supervised final answer.  The historical three-real-turn
shape remains available, while the five-real-turn shape is selected explicitly
for the formal five-tool-turn evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from agrinet.rag.distill.terminal_contract import final_answer_only_correction

TOOL_NAME = "agrinet_rag_search"
REJECTION_MESSAGE = "No additional retrieval is available. Use the evidence already returned."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tool_count(row: dict[str, Any]) -> int:
    return sum(message.get("role") == "tool_call" for message in row.get("messages") or [])


def _cell_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return "/".join(str(metadata.get(key) or "") for key in ("question_type", "language", "task_domain"))


def _token_proxy(row: dict[str, Any]) -> int:
    text = " ".join(
        str(message.get("content") or "")
        for message in row.get("messages") or []
        if isinstance(message, dict)
    )
    return max(1, (len(text) + 3) // 4)


def _synthetic_call() -> dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "arguments": {
            "retrieval_type": "visual",
            "query": "agricultural disease or pest visual features",
            "image": "query_image",
            "top_k": 3,
            "rationale": "Request one additional public retrieval after the available evidence.",
        },
    }


def _rejection_observation(metadata: dict[str, Any]) -> str:
    payload = {
        "source": "AgriNet public reference catalog",
        "retrieval_type": None,
        "query": None,
        "results": [],
        "status": "error",
        "error": "tool_budget_exhausted",
        "message": REJECTION_MESSAGE,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n" + final_answer_only_correction(metadata)


def derive(
    rows: list[dict[str, Any]], expected_hcv_tool_turns: int = 3, expected_hcv_rows: int = 32,
    one_call_terminal_limit: int | None = None,
    max_hcv_token_fraction: float = 1.0, min_direct_token_fraction: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if expected_hcv_tool_turns not in {3, 5}:
        raise ValueError("expected_hcv_tool_turns must be 3 or 5")
    if expected_hcv_rows < 8 or expected_hcv_rows % 8:
        raise ValueError("expected_hcv_rows must be a positive multiple of the eight task cells")
    if not 0.0 < max_hcv_token_fraction <= 1.0:
        raise ValueError("max_hcv_token_fraction must be in (0, 1]")
    if not 0.0 <= min_direct_token_fraction < 1.0:
        raise ValueError("min_direct_token_fraction must be in [0, 1)")
    one_call_sources = [row for row in rows if _tool_count(row) == 1]
    if one_call_terminal_limit is not None:
        if one_call_terminal_limit < 8 or one_call_terminal_limit % 8:
            raise ValueError("one_call_terminal_limit must be a multiple of the eight task cells")
        per_cell = one_call_terminal_limit // 8
        selected_one_call: list[dict[str, Any]] = []
        for cell in sorted({_cell_key(row) for row in one_call_sources}):
            candidates = sorted((row for row in one_call_sources if _cell_key(row) == cell), key=lambda row: str(row.get("sample_id") or ""))
            if len(candidates) < per_cell:
                raise ValueError(f"one-call terminal source shortage for {cell}: {len(candidates)} < {per_cell}")
            selected_one_call.extend(candidates[:per_cell])
        if len(selected_one_call) != one_call_terminal_limit:
            raise ValueError("one-call terminal source does not cover exactly eight balanced cells")
    else:
        selected_one_call = one_call_sources
    selected_ids = {str(row.get("sample_id") or "") for row in selected_one_call}
    variants: list[dict[str, Any]] = []
    source_counts: Counter[int] = Counter()
    for row in rows:
        tool_calls = _tool_count(row)
        if tool_calls not in {1, expected_hcv_tool_turns}:
            continue
        if tool_calls == 1 and str(row.get("sample_id") or "") not in selected_ids:
            continue
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
            raise ValueError(f"tool row lacks an assistant final: {row.get('sample_id')}")
        if "<answer>" not in str(messages[-1].get("content") or ""):
            raise ValueError(f"tool row lacks answer target: {row.get('sample_id')}")
        item = json.loads(json.dumps(row, ensure_ascii=False))
        metadata = dict(item.get("metadata") or {})
        base_id = str(item.get("sample_id") or "")
        item["sample_id"] = base_id + "--terminal-rejection-v2"
        metadata.update({
            "route": "terminal_rejection_v2",
            "terminal_rejection_source": base_id,
            "terminal_rejection_source_tool_turns": tool_calls,
            "terminal_policy": "public-tool-budget-rejection-v2",
            "student_user_contract": "agrinet.current-task-contract/v2",
        })
        item["metadata"] = metadata
        final_answer = item["messages"].pop()
        item["messages"].extend([
            {"role": "assistant", "content": "<think>Additional retrieval may help verify the remaining ambiguity.</think>"},
            {"role": "tool_call", "content": json.dumps(_synthetic_call(), ensure_ascii=False, separators=(",", ":"))},
            {"role": "tool", "content": _rejection_observation(metadata)},
            final_answer,
        ])
        variants.append(item)
        source_counts[tool_calls] += 1

    one_call_count = len(selected_one_call)
    expected_sources = Counter({1: one_call_count, expected_hcv_tool_turns: expected_hcv_rows})
    if not one_call_count:
        raise ValueError("source needs at least one one-call anchor")
    if source_counts != expected_sources:
        raise ValueError(
            f"expected 560 one-call and 32 HCV {expected_hcv_tool_turns}-call sources, got {dict(source_counts)}"
        )
    output = [*rows, *variants]
    ids = [str(row.get("sample_id") or "") for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("derived artifact has duplicate sample IDs")
    def role_valid(row: dict[str, Any]) -> bool:
        allowed = {
            ("system", "user"),
            ("user", "assistant"), ("assistant", "tool_call"),
            ("tool_call", "tool"), ("tool_call", "tool_response"),
            ("tool", "assistant"), ("tool_response", "assistant"),
        }
        messages = row.get("messages") or []
        return all((left.get("role"), right.get("role")) in allowed for left, right in zip(messages, messages[1:]))
    hcv_route = "hcv_three_query" if expected_hcv_tool_turns == 3 else "hcv_five_query"
    terminal_hcv_route = (
        "terminal_rejection_hcv_three_query"
        if expected_hcv_tool_turns == 3 else "terminal_rejection_hcv_five_query"
    )
    total_tokens = sum(_token_proxy(row) for row in output)
    hcv_tokens = sum(
        _token_proxy(row) for row in output
        if _tool_count(row) in {expected_hcv_tool_turns, expected_hcv_tool_turns + 1}
    )
    direct_tokens = sum(_token_proxy(row) for row in output if _tool_count(row) == 0)
    hcv_fraction = hcv_tokens / total_tokens if total_tokens else 1.0
    direct_fraction = direct_tokens / total_tokens if total_tokens else 0.0
    report = {
        "schema_version": "agrinet.hcv-sft-freeze/v5-terminal-rejection",
        "source_rows": len(rows), "rows": len(output), "terminal_rejection_rows": len(variants),
        "route_rows": {"direct_anchor": sum(_tool_count(row) == 0 for row in rows), "one_call_anchor": one_call_count, hcv_route: expected_hcv_rows, "terminal_rejection_one_call": one_call_count, terminal_hcv_route: expected_hcv_rows},
        "one_call_terminal_limit": one_call_terminal_limit,
        "route_token_proxy": {"total": total_tokens, "hcv": hcv_tokens, "hcv_fraction": hcv_fraction, "hcv_fraction_cap": max_hcv_token_fraction, "direct": direct_tokens, "direct_fraction": direct_fraction, "direct_fraction_floor": min_direct_token_fraction},
        "invariants": {
            "unique_sample_ids": True,
            "original_rows_unchanged_in_order": output[:len(rows)] == rows,
            "query_image_only": all(len(row.get("images") or []) == 1 for row in output),
            "public_label_blind_rejections": all("tool_budget_exhausted" in row["messages"][-2]["content"] for row in variants),
            "shared_terminal_contract": all(final_answer_only_correction(row["metadata"]) in row["messages"][-2]["content"] for row in variants),
            "swift_role_alternation": all(role_valid(row) for row in variants),
            "source_route_counts": source_counts == expected_sources,
            "hcv_terminal_shape": all(
                _tool_count(row) == expected_hcv_tool_turns + 1
                for row in variants
                if row["metadata"].get("terminal_rejection_source_tool_turns") == expected_hcv_tool_turns
            ),
            "hcv_token_weight_capped": hcv_fraction <= max_hcv_token_fraction,
            "direct_token_weight_floored": direct_fraction >= min_direct_token_fraction,
        },
    }
    report["training_authorized"] = all(report["invariants"].values())
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--artifact-id", default="agrinet-hcv-manual-json-v5-terminal-rejection")
    parser.add_argument("--expected-hcv-tool-turns", type=int, choices=(3, 5), default=3)
    parser.add_argument("--expected-hcv-rows", type=int, default=32)
    parser.add_argument("--one-call-terminal-limit", type=int, help="Balanced number of one-call terminal-rejection variants; defaults to all.")
    parser.add_argument("--max-hcv-token-fraction", type=float, default=1.0)
    parser.add_argument("--min-direct-token-fraction", type=float, default=0.0)
    args = parser.parse_args()
    if args.destination.exists() and any(args.destination.iterdir()):
        raise ValueError(f"destination already exists and is non-empty: {args.destination}")
    rows = read_jsonl(args.source)
    output, report = derive(rows, args.expected_hcv_tool_turns, args.expected_hcv_rows, args.one_call_terminal_limit, args.max_hcv_token_fraction, args.min_direct_token_fraction)
    args.destination.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in output).encode("utf-8")
    (args.destination / "data.jsonl").write_bytes(payload)
    report.update({
        "artifact_id": args.artifact_id, "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(payload).hexdigest(), "immutable": True,
    })
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (args.destination / "validation.json").write_text(text, encoding="utf-8")
    (args.destination / "manifest.yaml").write_text(text, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["training_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
