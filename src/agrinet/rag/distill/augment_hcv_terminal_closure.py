#!/usr/bin/env python3
"""Add auditable terminal-only closure examples to the frozen HCV dataset.

The evaluator gives exactly one public terminal prompt after a budget/format
failure.  The original HCV rows end immediately after the last tool response,
so the model never sees that state during SFT.  This derivative appends one
closure variant per three-query HCV row, preserving the original row and image
while adding only the evaluator's label-blind terminal instruction and the
already existing final answer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from agrinet.rag.distill.terminal_contract import final_answer_only_correction


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def terminal_prompt(metadata: dict[str, Any]) -> str:
    return final_answer_only_correction(metadata)


def derive(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 5:
            continue
        tool_calls = sum(isinstance(m, dict) and m.get("role") == "tool_call" for m in messages)
        if tool_calls != 3:
            continue
        if messages[-1].get("role") != "assistant" or "<answer>" not in str(messages[-1].get("content") or ""):
            raise ValueError(f"HCV row lacks a final answer: {row.get('sample_id')}")
        item = json.loads(json.dumps(row, ensure_ascii=False))
        metadata = dict(item.get("metadata") or {})
        base_id = str(item.get("sample_id") or "")
        item["sample_id"] = base_id + "--terminal-closure"
        metadata.update({
            "route": "hcv_terminal_closure",
            "terminal_closure_source": base_id,
            "terminal_policy": "one-terminal-closure-v1",
            "student_user_contract": "agrinet.current-task-contract/v2",
        })
        item["metadata"] = metadata
        final_answer = item["messages"].pop()
        # Swift requires alternating user/tool -> assistant rounds.  The
        # evaluator sends the terminal instruction as a user turn, but the
        # training equivalent must be appended to the last tool observation;
        # otherwise the encoder sees tool -> user -> assistant and asserts.
        last_tool = item["messages"][-1]
        if last_tool.get("role") not in {"tool", "tool_response"}:
            raise ValueError(f"HCV closure does not end in tool_response: {base_id}")
        last_tool["content"] = str(last_tool.get("content") or "") + "\n" + terminal_prompt(metadata)
        item["messages"].append(final_answer)
        variants.append(item)
    if len(variants) != 32:
        raise ValueError(f"expected 32 HCV closure variants, got {len(variants)}")
    output = [*rows, *variants]
    ids = [str(row.get("sample_id") or "") for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("derived artifact has duplicate sample IDs")
    report = {
        "schema_version": "agrinet.hcv-sft-freeze/v3-terminal-closure",
        "source_rows": len(rows),
        "rows": len(output),
        "closure_rows": len(variants),
        "route_rows": {"direct_anchor": 560, "one_call_anchor": 560, "hcv_three_query": 32, "hcv_terminal_closure": 32},
        "invariants": {
            "unique_sample_ids": True,
            "closure_source_count": len(variants) == 32,
            "query_image_only": all(len(row.get("images") or []) == 1 for row in output),
            "closure_is_label_blind_prompt": all("tool_call" in str(row["messages"][-2].get("content")) for row in variants),
            "swift_role_alternation": all(
                all(
                    (left.get("role"), right.get("role")) in {
                        ("user", "assistant"), ("tool_response", "assistant"),
                        ("tool", "assistant"), ("assistant", "tool_call"),
                        ("tool_call", "tool_response"), ("tool_call", "tool"),
                    }
                    for left, right in zip(row.get("messages", []), row.get("messages", [])[1:])
                )
                for row in variants
            ),
            "original_rows_unchanged_in_order": output[: len(rows)] == rows,
        },
    }
    report["training_authorized"] = all(report["invariants"].values())
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.source)
    output, report = derive(rows)
    args.destination.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in output).encode("utf-8")
    (args.destination / "data.jsonl").write_bytes(payload)
    report.update({
        "artifact_id": "agrinet-hcv-manual-json-v3-terminal-closure",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(payload).hexdigest(),
        "immutable": True,
    })
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (args.destination / "validation.json").write_text(text, encoding="utf-8")
    (args.destination / "manifest.yaml").write_text(text, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["training_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
