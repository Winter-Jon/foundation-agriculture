#!/usr/bin/env python3
"""Freeze the M3 anchor plus audited HCV two-turn trajectories."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from agrinet.data.rebuild_sft import canonical_json_hash, image_digest
from tools.rag_distill.catalog_and_isolation import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--hcv-accepted", type=Path, required=True)
    parser.add_argument("--hcv-audit", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--max-hcv-token-fraction", type=float, default=0.15)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def tool_turns(row: dict[str, Any]) -> int:
    return sum(message.get("role") == "tool_call" for message in row.get("messages") or [] if isinstance(message, dict))


def image_hash(row: dict[str, Any], root: Path | None = None) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    value = str(metadata.get("image_sha256") or "")
    if value:
        return value
    images = row.get("images") if isinstance(row.get("images"), list) else []
    if root is not None and images:
        try:
            return image_digest(str(images[0]), root)
        except (FileNotFoundError, OSError):
            return ""
    return ""


def token_proxy(row: dict[str, Any]) -> int:
    text = " ".join(str(message.get("content") or "") for message in row.get("messages") or [] if isinstance(message, dict))
    return max(1, (len(text) + 3) // 4)


def freeze(anchor: list[dict[str, Any]], hcv: list[dict[str, Any]], audit: dict[str, Any], max_hcv_fraction: float, root: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not audit.get("freeze_authorized"):
        raise ValueError("HCV post-collection audit does not authorize a freeze")
    if not 0.0 < max_hcv_fraction < 1.0:
        raise ValueError("max_hcv_token_fraction must be in (0, 1)")
    direct = [row for row in anchor if tool_turns(row) == 0]
    one_call = [row for row in anchor if tool_turns(row) == 1]
    other = [row for row in anchor if tool_turns(row) not in {0, 1}]
    if not direct or not one_call or other:
        raise ValueError("anchor must contain Direct and exactly-one-call RAG rows only")
    if len(hcv) != 32 or any(tool_turns(row) != 2 for row in hcv):
        raise ValueError("HCV input must be exactly 32 audited two-call rows")
    anchor_hashes = {image_hash(row, root) for row in anchor}
    hcv_hashes = {image_hash(row, root) for row in hcv}
    if not all(anchor_hashes) or not all(hcv_hashes) or anchor_hashes & hcv_hashes:
        raise ValueError("missing or overlapping anchor/HCV image hashes")
    hcv_tokens = sum(token_proxy(row) for row in hcv)
    total_tokens = sum(token_proxy(row) for row in [*anchor, *hcv])
    fraction = hcv_tokens / total_tokens if total_tokens else 1.0
    if fraction > max_hcv_fraction:
        raise ValueError(f"HCV token fraction {fraction:.4f} exceeds cap {max_hcv_fraction:.4f}")
    rows = [*anchor, *hcv]
    ids = [str(row.get("sample_id") or "") for row in rows]
    report = {
        "schema_version": "agrinet.hcv-sft-freeze/v1",
        "rows": len(rows),
        "route_rows": {"direct_anchor": len(direct), "one_call_anchor": len(one_call), "hcv_two_call": len(hcv)},
        "route_token_proxy": {"direct_anchor": sum(token_proxy(row) for row in direct), "one_call_anchor": sum(token_proxy(row) for row in one_call), "hcv_two_call": hcv_tokens, "hcv_fraction": fraction, "hcv_fraction_cap": max_hcv_fraction},
        "cell_counts": dict(sorted(Counter("/".join(str((row.get("metadata") or {}).get(key) or "") for key in ("question_type", "language", "task_domain")) for row in hcv).items())),
        "invariants": {"unique_sample_ids": len(ids) == len(set(ids)), "anchor_hcv_images_disjoint": not bool(anchor_hashes & hcv_hashes), "hcv_exactly_32_two_call": len(hcv) == 32 and all(tool_turns(row) == 2 for row in hcv), "direct_anchor_present": bool(direct), "one_call_anchor_present": bool(one_call), "hcv_token_weight_capped": fraction <= max_hcv_fraction},
    }
    report["training_authorized"] = all(report["invariants"].values())
    return rows, report


def main() -> int:
    args = parse_args()
    audit = json.loads(args.hcv_audit.read_text(encoding="utf-8"))
    rows, report = freeze(read_jsonl(args.anchor), read_jsonl(args.hcv_accepted), audit, args.max_hcv_token_fraction, args.repo_root)
    data_hash = canonical_json_hash(rows)
    if args.destination.exists():
        existing = args.destination / "data.jsonl"
        if not existing.is_file() or canonical_json_hash(read_jsonl(existing)) != data_hash:
            raise RuntimeError(f"immutable artifact conflict: {args.destination}")
    else:
        args.destination.mkdir(parents=True)
        write_jsonl(args.destination / "data.jsonl", rows)
    manifest = {"schema_version": "agrinet.hcv-sft-freeze/v1", "artifact_id": args.artifact_id, "immutable": True, "data_sha256": data_hash, "validation": report}
    (args.destination / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (args.destination / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(args.destination), "data_sha256": data_hash, "training_authorized": report["training_authorized"]}, ensure_ascii=False))
    return 0 if report["training_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
