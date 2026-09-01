#!/usr/bin/env python3
"""Build the immutable retention-aware v7 HCV training artifact.

This is intentionally a single fail-closed boundary between an independently
audited Micu collection and SFT.  It never contacts a teacher or retrieval
service.  The raw collection trace remains the evidence record; the emitted
student rows use the native compact public tool schema consumed by formal DP8
evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agrinet.data.rebuild_sft import canonical_json_hash
from agrinet.rag.distill.augment_hcv_terminal_rejection import derive
from agrinet.rag.distill.catalog_and_isolation import read_jsonl, write_jsonl
from agrinet.rag.distill.freeze_hcv_sft import freeze, tool_turns

HCV_ROWS = 64
DIRECT_REPLAY_FACTOR = 4
ONE_CALL_TERMINAL_LIMIT = 64
MAX_HCV_TOKEN_FRACTION = 0.52
MIN_DIRECT_TOKEN_FRACTION = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--hcv-accepted", type=Path, required=True)
    parser.add_argument("--hcv-audit", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--artifact-id", default="agrinet-hcv-manual-json-v7-five-turn-retention-aware")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def build(
    anchor: list[dict[str, Any]], hcv: list[dict[str, Any]], audit: dict[str, Any], root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if audit.get("freeze_authorized") is not True:
        raise ValueError("post-collection private audit does not authorize the v7 freeze")
    anchors = [row for row in anchor if tool_turns(row) in {0, 1}]
    direct_count = sum(tool_turns(row) == 0 for row in anchors)
    one_call_count = sum(tool_turns(row) == 1 for row in anchors)
    if (direct_count, one_call_count) != (560, 560):
        raise ValueError(f"v7 anchor must contain exactly 560 Direct and 560 one-call rows, got {(direct_count, one_call_count)}")
    frozen, freeze_report = freeze(
        anchors, hcv, audit, MAX_HCV_TOKEN_FRACTION, root,
        expected_hcv_tool_turns=5, expected_hcv_rows=HCV_ROWS,
        require_hcv_cell_balance=True, direct_replay_factor=DIRECT_REPLAY_FACTOR,
        min_direct_token_fraction=MIN_DIRECT_TOKEN_FRACTION,
    )
    final_rows, terminal_report = derive(
        frozen, expected_hcv_tool_turns=5, expected_hcv_rows=HCV_ROWS,
        one_call_terminal_limit=ONE_CALL_TERMINAL_LIMIT,
        max_hcv_token_fraction=MAX_HCV_TOKEN_FRACTION,
        min_direct_token_fraction=MIN_DIRECT_TOKEN_FRACTION,
    )
    validation = {
        "schema_version": "agrinet.hcv-v7-retention-freeze/v1",
        "hcv_audit": audit,
        "freeze": freeze_report,
        "terminal_rejection": terminal_report,
        "parameters": {
            "hcv_rows": HCV_ROWS, "direct_replay_factor": DIRECT_REPLAY_FACTOR,
            "one_call_terminal_limit": ONE_CALL_TERMINAL_LIMIT,
            "max_hcv_token_fraction": MAX_HCV_TOKEN_FRACTION,
            "min_direct_token_fraction": MIN_DIRECT_TOKEN_FRACTION,
        },
        "invariants": {
            "private_audit_authorized": audit.get("freeze_authorized") is True,
            "freeze_authorized": freeze_report.get("training_authorized") is True,
            "terminal_authorized": terminal_report.get("training_authorized") is True,
            "all_native_tool_roles": all(
                message.get("role") != "tool_response"
                for row in final_rows for message in row.get("messages") or []
            ),
        },
    }
    validation["training_authorized"] = all(validation["invariants"].values())
    if not validation["training_authorized"]:
        raise ValueError("v7 retention freeze invariants failed")
    return final_rows, validation


def main() -> int:
    args = parse_args()
    rows, validation = build(
        read_jsonl(args.anchor), read_jsonl(args.hcv_accepted),
        json.loads(args.hcv_audit.read_text(encoding="utf-8")), args.repo_root,
    )
    payload_hash = canonical_json_hash(rows)
    if args.destination.exists():
        data_path = args.destination / "data.jsonl"
        if not data_path.is_file() or canonical_json_hash(read_jsonl(data_path)) != payload_hash:
            raise RuntimeError(f"immutable artifact conflict: {args.destination}")
    else:
        args.destination.mkdir(parents=True)
        write_jsonl(args.destination / "data.jsonl", rows)
    manifest = {
        "schema_version": "agrinet.hcv-v7-retention-freeze/v1", "artifact_id": args.artifact_id,
        "immutable": True, "data_sha256": payload_hash, "validation": validation,
    }
    (args.destination / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.destination / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(json.dumps({"artifact": str(args.destination), "data_sha256": payload_hash, "training_authorized": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
