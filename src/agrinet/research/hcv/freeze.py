#!/usr/bin/env python3
"""Freeze M3 anchors plus audited HCV trajectories.

The historical three-query route remains the default for reproducibility.  The
five-query mode is explicitly selected only for the formal five-tool-turn HCV
route, so an audited trajectory cannot be silently frozen under a different
budget contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

SCRIPT_ROOT = Path(__file__).resolve().parents[4]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from agrinet.data.rebuild_sft import canonical_json_hash, image_digest
from agrinet.research.shared.catalog import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--hcv-accepted", type=Path, required=True)
    parser.add_argument("--hcv-audit", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--expected-hcv-tool-turns", type=int, choices=(3, 5), default=3)
    parser.add_argument(
        "--expected-hcv-rows", type=int, default=32,
        help="Exact audited HCV trajectory count. It must be balanced across all eight task cells.",
    )
    parser.add_argument(
        "--require-hcv-cell-balance", action="store_true",
        help="Require exactly --expected-hcv-rows/8 trajectories in each task cell (required for new freezes).",
    )
    parser.add_argument(
        "--direct-replay-factor", type=int, default=1,
        help="Repeat immutable Direct anchors with distinct provenance IDs to protect Direct retention.",
    )
    parser.add_argument(
        "--min-direct-token-fraction", type=float, default=0.0,
        help="Fail the freeze unless Direct anchors occupy at least this fraction of serialized training tokens.",
    )
    parser.add_argument(
        "--extract-anchor-direct-and-one-call", action="store_true",
        help="Select only 0/1-call rows from a historical mixed source and require 560 of each.",
    )
    parser.add_argument("--max-hcv-token-fraction", type=float, default=0.50)
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


def route_qualified_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make paired anchor route identities explicit without changing content."""
    seen: set[str] = set()
    counts: dict[str, int] = {}
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        sample_id = str(item.get("sample_id") or "")
        occurrence = counts.get(sample_id, 0)
        counts[sample_id] = occurrence + 1
        if sample_id in seen:
            turns = tool_turns(item)
            suffix = "one-call-anchor" if turns == 1 else f"route-duplicate-{occurrence}"
            item["sample_id"] = f"{sample_id}--{suffix}"
        if str(item.get("sample_id") or "") in seen:
            raise ValueError(f"unresolvable duplicate sample_id: {sample_id}")
        seen.add(str(item.get("sample_id") or ""))
        output.append(item)
    return output


def query_only_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the queried image; retrieved references remain text evidence."""
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        images = item.get("images") if isinstance(item.get("images"), list) else []
        if not images:
            raise ValueError(f"training row has no query image: {item.get('sample_id')}")
        metadata = dict(item.get("metadata") or {})
        metadata["reference_image_count"] = max(len(images) - 1, 0)
        item["metadata"] = metadata
        item["images"] = [images[0]]
        output.append(item)
    return output


def canonical_tool_observation(content: str) -> str:
    """Reduce a teacher trace to the exact public evidence schema used at eval.

    Raw Micu traces preserve descriptions, source identifiers, and reference
    image handles for post-collection audit.  Those fields are not visible to
    the student at formal evaluation, where the native ``tool`` turn contains
    only the compact public candidate card.  Keeping them in SFT would both
    create a wire-format mismatch and make five-turn rows dominate the token
    mixture for reasons unrelated to the HCV policy.
    """
    decoder = json.JSONDecoder()
    try:
        payload, consumed = decoder.raw_decode(content.lstrip())
    except json.JSONDecodeError as exc:
        raise ValueError("tool observation is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("tool observation must be a JSON object")
    results: list[dict[str, Any]] = []
    for hit in payload.get("results") or []:
        if not isinstance(hit, dict):
            continue
        item = {
            "rank": hit.get("rank"),
            "name": hit.get("name", hit.get("class_name")),
            "name_zh": hit.get("name_zh", hit.get("chinese_name")),
            "similarity": hit.get("similarity", hit.get("score")),
        }
        similar = hit.get("similar_classes")
        if isinstance(similar, list):
            cards = [
                {"name": value.get("name"), "name_zh": value.get("name_zh")}
                for value in similar[:5]
                if isinstance(value, dict) and (value.get("name") or value.get("name_zh"))
            ]
            if cards:
                item["similar_classes"] = cards
        results.append(item)
    compact = {
        "source": "AgriNet public reference catalog",
        "retrieval_type": payload.get("retrieval_type"),
        "query": payload.get("query"),
        "results": results,
        "status": payload.get("status"),
    }
    for key in ("error", "message", "errors"):
        if key in payload:
            compact[key] = payload[key]
    # Terminal-rejection SFT intentionally appends the shared public
    # answer-only correction after the JSON tool observation. Keep that exact
    # label-blind state while compacting only the evidence JSON.
    suffix = content.lstrip()[consumed:].strip()
    rendered = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    return f"{rendered}\n{suffix}" if suffix else rendered


def canonicalize_tool_schema(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use native tool roles and evaluator-equivalent public payloads."""
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        messages: list[dict[str, Any]] = []
        for message in item.get("messages") or []:
            rendered = dict(message)
            if rendered.get("role") in {"tool", "tool_response"}:
                rendered["role"] = "tool"
                rendered["content"] = canonical_tool_observation(str(rendered.get("content") or ""))
            messages.append(rendered)
        item["messages"] = messages
        metadata = dict(item.get("metadata") or {})
        metadata["tool_observation_schema"] = "agrinet.eval-native-public-card/v1"
        item["metadata"] = metadata
        output.append(item)
    return output


def replay_direct_anchors(rows: list[dict[str, Any]], factor: int) -> list[dict[str, Any]]:
    """Repeat Direct supervision transparently, never pretending repeats are new images."""
    if factor < 1:
        raise ValueError("direct_replay_factor must be positive")
    output: list[dict[str, Any]] = []
    for row in rows:
        for replica in range(factor):
            item = json.loads(json.dumps(row, ensure_ascii=False))
            if replica:
                source_id = str(item.get("sample_id") or "")
                item["sample_id"] = f"{source_id}--direct-replay-{replica + 1}"
                metadata = dict(item.get("metadata") or {})
                metadata.update({"direct_replay_source": source_id, "direct_replay_index": replica + 1})
                item["metadata"] = metadata
            output.append(item)
    return output


def freeze(
    anchor: list[dict[str, Any]],
    hcv: list[dict[str, Any]],
    audit: dict[str, Any],
    max_hcv_fraction: float,
    root: Path | None = None,
    expected_hcv_tool_turns: int = 3,
    expected_hcv_rows: int = 32,
    require_hcv_cell_balance: bool = False,
    direct_replay_factor: int = 1,
    min_direct_token_fraction: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not audit.get("freeze_authorized"):
        raise ValueError("HCV post-collection audit does not authorize a freeze")
    if not 0.0 < max_hcv_fraction < 1.0:
        raise ValueError("max_hcv_token_fraction must be in (0, 1)")
    if expected_hcv_tool_turns not in {3, 5}:
        raise ValueError("expected_hcv_tool_turns must be 3 or 5")
    if expected_hcv_rows < 1 or (require_hcv_cell_balance and expected_hcv_rows % 8):
        raise ValueError("expected_hcv_rows must be a positive multiple of the eight task cells")
    if not 0.0 <= min_direct_token_fraction < 1.0:
        raise ValueError("min_direct_token_fraction must be in [0, 1)")
    direct = [row for row in anchor if tool_turns(row) == 0]
    one_call = [row for row in anchor if tool_turns(row) == 1]
    other = [row for row in anchor if tool_turns(row) not in {0, 1}]
    if not direct or not one_call or other:
        raise ValueError("anchor must contain Direct and exactly-one-call RAG rows only")
    if len(hcv) != expected_hcv_rows or any(tool_turns(row) != expected_hcv_tool_turns for row in hcv):
        raise ValueError(f"HCV input must be exactly {expected_hcv_rows} audited {expected_hcv_tool_turns}-call rows")
    anchor_hashes = {image_hash(row, root) for row in anchor}
    hcv_hashes = {image_hash(row, root) for row in hcv}
    if not all(anchor_hashes) or not all(hcv_hashes) or anchor_hashes & hcv_hashes:
        raise ValueError("missing or overlapping anchor/HCV image hashes")
    direct_replayed = replay_direct_anchors(direct, direct_replay_factor)
    rows = query_only_rows(canonicalize_tool_schema(route_qualified_rows([*direct_replayed, *one_call, *hcv])))
    # Measure the actual serialized student data, after the evaluator-equivalent
    # card conversion. Raw teacher trace length is retained for audit only and
    # must never decide the SFT mixture.
    hcv_tokens = sum(token_proxy(row) for row in rows if tool_turns(row) == expected_hcv_tool_turns)
    total_tokens = sum(token_proxy(row) for row in rows)
    fraction = hcv_tokens / total_tokens if total_tokens else 1.0
    direct_tokens = sum(token_proxy(row) for row in rows if tool_turns(row) == 0)
    direct_fraction = direct_tokens / total_tokens if total_tokens else 0.0
    if fraction > max_hcv_fraction:
        raise ValueError(f"HCV token fraction {fraction:.4f} exceeds cap {max_hcv_fraction:.4f}")
    if direct_fraction < min_direct_token_fraction:
        raise ValueError(f"Direct token fraction {direct_fraction:.4f} is below floor {min_direct_token_fraction:.4f}")
    ids = [str(row.get("sample_id") or "") for row in rows]
    hcv_route = "hcv_three_query" if expected_hcv_tool_turns == 3 else "hcv_five_query"
    hcv_invariant = f"hcv_exactly_{expected_hcv_rows}_{expected_hcv_tool_turns}_query"
    hcv_cell_counts = Counter(
        "/".join(str((row.get("metadata") or {}).get(key) or "") for key in ("question_type", "language", "task_domain"))
        for row in hcv
    )
    expected_cell_count = expected_hcv_rows // 8
    route_rows = {"direct_anchor": len(direct), "one_call_anchor": len(one_call), hcv_route: len(hcv)}
    if direct_replay_factor > 1:
        route_rows["direct_replayed"] = len(direct_replayed)
    report = {
        "schema_version": "agrinet.hcv-sft-freeze/v1",
        "rows": len(rows),
        "route_rows": route_rows,
        "route_token_proxy": {"direct_anchor": direct_tokens, "direct_replay_factor": direct_replay_factor, "direct_fraction": direct_fraction, "direct_fraction_floor": min_direct_token_fraction, "one_call_anchor": sum(token_proxy(row) for row in rows if tool_turns(row) == 1), hcv_route: hcv_tokens, "hcv_fraction": fraction, "hcv_fraction_cap": max_hcv_fraction},
        "cell_counts": dict(sorted(hcv_cell_counts.items())),
        "expected_hcv_rows": expected_hcv_rows,
        "expected_hcv_rows_per_cell": expected_cell_count,
        "invariants": {"unique_sample_ids": len(ids) == len(set(ids)), "query_image_only": all(len(row.get("images") or []) == 1 for row in rows), "native_eval_tool_schema": all(message.get("role") != "tool_response" for row in rows for message in row.get("messages") or []), "anchor_hcv_images_disjoint": not bool(anchor_hashes & hcv_hashes), hcv_invariant: len(hcv) == expected_hcv_rows and all(tool_turns(row) == expected_hcv_tool_turns for row in hcv), "hcv_cells_balanced": not require_hcv_cell_balance or (len(hcv_cell_counts) == 8 and all(value == expected_cell_count for value in hcv_cell_counts.values())), "direct_anchor_present": bool(direct), "one_call_anchor_present": bool(one_call), "direct_token_weight_floored": direct_fraction >= min_direct_token_fraction, "hcv_token_weight_capped": fraction <= max_hcv_fraction},
    }
    report["training_authorized"] = all(report["invariants"].values())
    return rows, report


def main() -> int:
    args = parse_args()
    audit = json.loads(args.hcv_audit.read_text(encoding="utf-8"))
    anchor = read_jsonl(args.anchor)
    if args.extract_anchor_direct_and_one_call:
        anchor = [row for row in anchor if tool_turns(row) in {0, 1}]
        counts = Counter(tool_turns(row) for row in anchor)
        if counts != Counter({0: 560, 1: 560}):
            raise ValueError(f"mixed anchor source must yield 560 Direct and 560 one-call rows, got {dict(counts)}")
    rows, report = freeze(
        anchor, read_jsonl(args.hcv_accepted), audit, args.max_hcv_token_fraction,
        args.repo_root, args.expected_hcv_tool_turns, args.expected_hcv_rows, args.require_hcv_cell_balance, args.direct_replay_factor, args.min_direct_token_fraction,
    )
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
