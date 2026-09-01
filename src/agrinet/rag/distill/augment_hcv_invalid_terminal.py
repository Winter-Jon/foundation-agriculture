#!/usr/bin/env python3
"""Add balanced, valid terminal-closure trajectories to HCV SFT data.

The former v8 augmentation supervised malformed JSON and is retired. A valid
training representation must end in an assistant answer and use the native
assistant -> tool_call -> tool -> assistant alternation. The added trajectory
therefore models a public, label-blind budget rejection after a canonical
follow-up request; it never places malformed JSON or a user turn after the
assistant response.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agrinet.rag.distill.terminal_contract import final_answer_only_correction


def _cell(row: dict[str, Any]) -> str:
    m = row.get("metadata") or {}
    return "/".join(str(m.get(k) or "") for k in ("question_type", "language", "task_domain"))


def _tool_calls(row: dict[str, Any]) -> int:
    return sum(m.get("role") == "tool_call" for m in row.get("messages") or [])


def build(rows: list[dict[str, Any]], per_cell: int = 8, *, route_name: str = "terminal_context_v3", sample_suffix: str = "--terminal-context-v3") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = [
        r for r in rows
        if _tool_calls(r) == 1
        and (r.get("metadata") or {}).get("route") not in {
            "invalid_terminal_v1", "invalid_terminal_context_v2",
            "terminal_context_v3", "terminal_rejection_v2",
        }
    ]
    cells = sorted({_cell(r) for r in sources})
    if len(cells) != 8:
        raise ValueError(f"expected eight task cells, got {cells}")
    selected: list[dict[str, Any]] = []
    for cell in cells:
        candidates = sorted((r for r in sources if _cell(r) == cell), key=lambda r: str(r.get("sample_id") or ""))
        if len(candidates) < per_cell:
            raise ValueError(f"source shortage for {cell}: {len(candidates)} < {per_cell}")
        selected.extend(candidates[:per_cell])
    variants: list[dict[str, Any]] = []
    for source in selected:
        item = json.loads(json.dumps(source, ensure_ascii=False))
        messages = item["messages"]
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"source lacks final assistant: {source.get('sample_id')}")
        final = messages.pop()
        metadata = dict(item.get("metadata") or {})
        base_id = str(item.get("sample_id") or "")
        item["sample_id"] = base_id + sample_suffix
        metadata.update({
            "route": route_name,
            "invalid_terminal_source": base_id,
            "terminal_policy": "public-tool-budget-rejection-v3",
        })
        item["metadata"] = metadata
        public_error = {
            "source": "AgriNet public reference catalog", "retrieval_type": None,
            "query": None, "results": [], "status": "error",
            "error": "tool_budget_exhausted",
            "message": "No additional retrieval is available. Use the evidence already returned.",
        }
        correction = (
            "Tool budget exhausted: do not call any more tools. Give the final answer now. "
            "Use the trained final form <think>brief evidence</think><answer>...</answer> and answer only from public evidence."
            if metadata.get("language") != "zh" else
            "工具调用次数已用尽：不要再调用工具。现在使用训练时的最终格式 <think>简短证据</think><answer>...</answer>，只根据公开证据回答。"
        )
        follow_up = {
            "name": "agrinet_rag_search",
            "arguments": {
                "query": "one additional public retrieval to verify the remaining ambiguity",
                "retrieval_type": "visual", "image": "query_image",
                "top_k": 3, "rationale": "Request one final public comparison before answering.",
            },
        }
        messages.extend([
            {"role": "assistant", "content": "<think>Additional retrieval may help verify the remaining ambiguity.</think>"},
            {"role": "tool_call", "content": json.dumps(follow_up, ensure_ascii=False, separators=(",", ":"))},
            {"role": "tool", "content": json.dumps(public_error, ensure_ascii=False, separators=(",", ":")) + "\n" + correction},
            final,
        ])
        variants.append(item)
    output = [*rows, *variants]
    ids = [str(r.get("sample_id") or "") for r in output]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs after invalid-terminal augmentation")
    report = {
        "schema_version": "agrinet.hcv-terminal-context/v3",
        "source_rows": len(rows), "invalid_terminal_rows": len(variants),
        "cell_counts": {cell: sum(_cell(r) == cell for r in variants) for cell in cells},
        "invariants": {
            "unique_sample_ids": True, "balanced_eight_cells": all(sum(_cell(r) == cell for r in variants) == per_cell for cell in cells),
            "query_image_only": all(len(r.get("images") or []) == 1 for r in variants),
            "private_label_absent_from_added_messages": all("final_label" not in json.dumps(r["messages"], ensure_ascii=False) for r in variants),
            "native_roles": all(m.get("role") != "tool_response" for r in variants for m in r.get("messages") or []),
            "no_invalid_assistant_supervision": all(
                not (m.get("role") == "assistant" and "invalid_tool_call" in str(m.get("content") or ""))
                for r in variants for m in r.get("messages") or []
            ),
            "assistant_final": all((r.get("messages") or [])[-1].get("role") == "assistant" for r in variants),
            "swift_role_alternation": all(
                (left.get("role"), right.get("role")) in {
                    ("system", "user"), ("user", "assistant"),
                    ("user", "tool_call"),
                    ("assistant", "tool_call"), ("tool_call", "tool"),
                    ("tool", "assistant"),
                }
                for r in variants for left, right in zip(r.get("messages") or [], (r.get("messages") or [])[1:])
            ),
        },
    }
    report["training_authorized"] = all(report["invariants"].values())
    if not report["training_authorized"]:
        raise ValueError("invalid terminal augmentation invariants failed")
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(x) for x in args.input.read_text(encoding="utf-8").splitlines() if x.strip()]
    output, report = build(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in output), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(output), "report": str(args.report), "training_authorized": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
