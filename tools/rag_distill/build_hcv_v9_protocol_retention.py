#!/usr/bin/env python3
"""Freeze the v9 HCV data: safe terminal context plus Direct retention replay.

v8 accidentally supervised a truncated JSON request as assistant content.  v9
starts from the immutable v7 freeze, adds only context-only invalid-tool states,
then makes the additional Direct replay explicit.  The resulting artifact is a
new immutable training boundary; it never modifies v7/v8.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agrinet.data.rebuild_sft import canonical_json_hash
from tools.rag_distill.augment_hcv_invalid_terminal import build as add_invalid_context
from tools.rag_distill.catalog_and_isolation import read_jsonl, write_jsonl
from tools.rag_distill.freeze_hcv_sft import token_proxy, tool_turns


def _replay_direct(rows: list[dict[str, Any]], factor: int) -> list[dict[str, Any]]:
    if factor < 1:
        raise ValueError("direct replay factor must be positive")
    direct = [row for row in rows if tool_turns(row) == 0]
    if not direct:
        raise ValueError("source contains no Direct anchors")
    output: list[dict[str, Any]] = []
    for row in direct:
        for index in range(factor):
            item = json.loads(json.dumps(row, ensure_ascii=False))
            source_id = str(item.get("sample_id") or "")
            item["sample_id"] = f"{source_id}--v9-direct-retention-{index + 1}"
            metadata = dict(item.get("metadata") or {})
            metadata.update({
                "route": "v9_direct_retention_replay",
                "direct_retention_source": source_id,
                "direct_retention_index": index + 1,
            })
            item["metadata"] = metadata
            output.append(item)
    return output


def build(rows: list[dict[str, Any]], direct_replay_factor: int = 1, min_direct_token_fraction: float = 0.35) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contextualized, invalid_report = add_invalid_context(rows)
    direct_replays = _replay_direct(contextualized, direct_replay_factor)
    output = [*contextualized, *direct_replays]
    ids = [str(row.get("sample_id") or "") for row in output]
    total_tokens = sum(token_proxy(row) for row in output)
    direct_tokens = sum(token_proxy(row) for row in output if tool_turns(row) == 0)
    direct_fraction = direct_tokens / total_tokens if total_tokens else 0.0
    report = {
        "schema_version": "agrinet.hcv-v9-protocol-retention/v1",
        "source_rows": len(rows),
        "rows": len(output),
        "invalid_terminal_context": invalid_report,
        "direct_retention_replays": len(direct_replays),
        "route_token_proxy": {
            "total": total_tokens,
            "direct": direct_tokens,
            "direct_fraction": direct_fraction,
            "direct_fraction_floor": min_direct_token_fraction,
        },
        "invariants": {
            "source_unchanged_in_order": output[:len(rows)] == rows,
            "unique_sample_ids": len(ids) == len(set(ids)),
            "query_image_only": all(len(row.get("images") or []) == 1 for row in output),
            "native_roles": all(message.get("role") != "tool_response" for row in output for message in row.get("messages") or []),
            "no_invalid_assistant_supervision": all(
                not (message.get("role") == "assistant" and "invalid_tool_call" in str(message.get("content") or ""))
                for row in output for message in row.get("messages") or []
            ),
            "direct_token_weight_floored": direct_fraction >= min_direct_token_fraction,
            "invalid_context_authorized": invalid_report.get("training_authorized") is True,
        },
    }
    report["training_authorized"] = all(report["invariants"].values())
    if not report["training_authorized"]:
        raise ValueError("v9 protocol-retention invariants failed")
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--artifact-id", default="agrinet-hcv-manual-json-v9-protocol-retention")
    parser.add_argument("--direct-replay-factor", type=int, default=1)
    parser.add_argument("--min-direct-token-fraction", type=float, default=0.35)
    args = parser.parse_args()
    if args.destination.exists() and any(args.destination.iterdir()):
        raise ValueError(f"destination already exists and is non-empty: {args.destination}")
    rows, report = build(read_jsonl(args.source), args.direct_replay_factor, args.min_direct_token_fraction)
    args.destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.destination / "data.jsonl", rows)
    payload_hash = canonical_json_hash(rows)
    manifest = {
        "schema_version": "agrinet.hcv-v9-protocol-retention/v1",
        "artifact_id": args.artifact_id, "immutable": True,
        "data_sha256": payload_hash, "validation": report,
    }
    (args.destination / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.destination / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(json.dumps({"artifact": str(args.destination), "data_sha256": payload_hash, "training_authorized": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
