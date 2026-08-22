#!/usr/bin/env python3
"""Measure public-evidence delta for image-isolated HCV retrieval actions.

This is a retrieval-only diagnostic: labels are retained solely after all calls
for offline truth-hit auditing.  They are never incorporated into a query, an
HTTP request, or a model-visible tool response.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from agrinet.data.rebuild_sft import CELLS, image_digest
from tools.rag_distill.build_reconstructive_direct_candidates import isolation_hashes
from tools.rag_distill.catalog_and_isolation import ROOT, read_jsonl, write_jsonl

SOURCE = ROOT / "outputs/vlm_data/disease_pest_large/contrast_samples_vit_base.jsonl"
SEED = "hcv-preflight-v1-20260822"
FORMAL_618 = ROOT / "outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl"
CURRENT_SFT_SOURCES = (
    ROOT / "outputs/artifacts/datasets/agrinet-hermes-long-direct-blind-rag-1to1-v2/sft-hermes-current-contract-v3/data.jsonl",
    ROOT / "outputs/artifacts/datasets/agrinet-hermes-long-direct-blind-rag-1to1-v2/sft-hermes-native-json-current-contract-v4/data.jsonl",
    ROOT / "outputs/artifacts/datasets/agrinet-hermes-long-direct-blind-rag-1to1-v2/pilot-views/b2-m2-native-json-current-contract-direct3-rag1-v4/data.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rag-api", default="http://127.0.0.1:8077")
    parser.add_argument("--per-cell", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--expand-top-k", type=int, default=10, help="wider second visual candidate budget; must exceed --top-k")
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def _digest_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(str(row.get("sample_id") or "").encode()).hexdigest()


def _row_image(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    images = row.get("images") if isinstance(row.get("images"), list) else []
    return str(row.get("query_image") or row.get("image_path") or metadata.get("query_image") or metadata.get("image_path") or (images[0] if images else ""))


def explicit_isolation_hashes(root: Path = ROOT) -> tuple[set[str], dict[str, int]]:
    """Hash the fixed formal/SFT sources omitted by broad historical scans.

    The broad legacy isolation helper remains the first defence.  These named
    sources are added so Phase-1 has an auditable proof for the current formal
    618 manifest and the exact SFT derivatives that seeded M3.
    """
    hashes: set[str] = set()
    counts: dict[str, int] = {}
    for path in (FORMAL_618, *CURRENT_SFT_SOURCES):
        count = 0
        if path.is_file():
            for item in read_jsonl(path):
                image = _row_image(item)
                if not image:
                    continue
                try:
                    hashes.add(image_digest(image, root))
                    count += 1
                except (FileNotFoundError, OSError):
                    continue
        counts[str(path.relative_to(root))] = count
    return hashes, counts


def select_rows(source: list[dict[str, Any]], per_cell: int, root: Path = ROOT) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    historical_forbidden = isolation_hashes(root)
    explicit_forbidden, explicit_sources = explicit_isolation_hashes(root)
    forbidden = historical_forbidden | explicit_forbidden
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        image = str(row.get("query_image") or "")
        if not image:
            continue
        try:
            digest = image_digest(image, root)
        except (FileNotFoundError, OSError):
            continue
        if digest in forbidden:
            continue
        domain = str(row.get("task_domain") or "")
        if domain not in {"disease", "pest"}:
            continue
        item = dict(row)
        item["image_sha256"] = digest
        pools[domain].append(item)

    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()
    shortages: dict[str, int] = {}
    for question_type, language, domain in CELLS:
        key = "/".join((question_type, language, domain))
        ordered = sorted(pools[domain], key=lambda row: hashlib.sha256(f"{SEED}:{key}:{_digest_key(row)}".encode()).hexdigest())
        cell = []
        for row in ordered:
            image = str(row["query_image"])
            if image in used_images:
                continue
            used_images.add(image)
            cell.append({
                "id": f"hcv-preflight-{len(selected) + len(cell) + 1:03d}",
                "query_image": image,
                "image_sha256": row["image_sha256"],
                "question_type": question_type,
                "language": language,
                "task_domain": domain,
                # Audit-only, never supplied to retrieve().
                "audit_truth_code": str(row.get("final_label") or ""),
                "source_sample_id": str(row.get("sample_id") or ""),
            })
            if len(cell) == per_cell:
                break
        selected.extend(cell)
        shortages[key] = max(0, per_cell - len(cell))
    audit = {
        "forbidden_hashes": len(forbidden),
        "historical_forbidden_hashes": len(historical_forbidden),
        "explicit_forbidden_hashes": len(explicit_forbidden),
        "explicit_source_rows_hashed": explicit_sources,
    }
    return selected, shortages, audit


def retrieve(api: str, retrieval_type: str, *, image: str | None, text: str, top_k: int, timeout: int) -> dict[str, Any]:
    body: dict[str, Any] = {"top_k": top_k}
    if text:
        body["text"] = text
    if image is not None:
        body["image_path"] = image
    response = requests.post(f"{api.rstrip('/')}/search/{retrieval_type}", json=body, timeout=timeout)
    response.raise_for_status()
    raw = response.json()
    hits = raw.get("hybrid") or []
    return {"retrieval_type": retrieval_type, "request": body, "results": hits[:top_k]}


def names(results: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for result in results:
        for key in ("english_name", "chinese_name"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
        for value in result.get("alias_en") or []:
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
        for value in result.get("similar_english_classes") or []:
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
    return list(dict.fromkeys(values))


def first_public_followup(first: dict[str, Any]) -> tuple[str, str] | None:
    hits = first.get("results") or []
    if not hits:
        return None
    top = hits[0]
    anchor = str(top.get("english_name") or "").strip()
    neighbors = [str(value).strip() for value in (top.get("similar_english_classes") or []) if str(value).strip()]
    if not anchor or not neighbors:
        return None
    # Public names from the first response, not a sample label.
    return anchor, neighbors[0]


def codes(results: list[dict[str, Any]]) -> set[str]:
    return {str(result.get("code") or "") for result in results if str(result.get("code") or "")}


def run_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    image = str(row["query_image"])
    first = retrieve(args.rag_api, "visual", image=image, text="visible agricultural symptoms", top_k=args.top_k, timeout=args.timeout)
    followup = first_public_followup(first)
    actions: dict[str, dict[str, Any]] = {"visual_first": first}
    errors: list[str] = []
    # This is a distinct, budgeted evidence request rather than an exact
    # repeat: the model first sees a compact hypothesis set and then asks for
    # additional visual candidates when that set cannot resolve the case.
    actions["visual_expand"] = retrieve(args.rag_api, "visual", image=image, text="visible agricultural symptoms", top_k=args.expand_top_k, timeout=args.timeout)
    if followup is not None:
        anchor, neighbor = followup
        comparison = f"compare {anchor} and {neighbor} symptoms"
        actions["balanced_compare"] = retrieve(args.rag_api, "balanced", image=image, text=comparison, top_k=args.top_k, timeout=args.timeout)
        actions["semantic_compare"] = retrieve(args.rag_api, "semantic", image=None, text=comparison, top_k=args.top_k, timeout=args.timeout)
        actions["name_confirm"] = retrieve(args.rag_api, "name", image=None, text=neighbor, top_k=args.top_k, timeout=args.timeout)
    else:
        errors.append("missing_public_anchor_or_similar_class_for_followup")
    truth = str(row["audit_truth_code"])
    first_codes = codes(first["results"])
    actions_audit = {}
    for name, action in actions.items():
        result_codes = codes(action["results"])
        actions_audit[name] = {
            "truth_hit": truth in result_codes,
            "codes": sorted(result_codes),
            "new_codes_vs_first": sorted(result_codes - first_codes),
            "evidence_changed": result_codes != first_codes,
        }
    return {**row, "actions": actions, "audit": {"first_truth_hit": truth in first_codes, "actions": actions_audit, "errors": errors}}


def summarize(rows: list[dict[str, Any]], shortages: dict[str, int]) -> dict[str, Any]:
    actions = ("visual_expand", "balanced_compare", "semantic_compare", "name_confirm")
    summary: dict[str, Any] = {"rows": len(rows), "shortages": shortages, "by_action": {}, "by_cell": {}, "errors": []}
    for action in actions:
        applicable = [row for row in rows if action in row.get("actions", {})]
        changed = [row for row in applicable if row["audit"]["actions"][action]["evidence_changed"]]
        repaired = [row for row in applicable if not row["audit"]["first_truth_hit"] and row["audit"]["actions"][action]["truth_hit"]]
        summary["by_action"][action] = {
            "applicable": len(applicable),
            "evidence_changed": len(changed),
            "evidence_change_rate": len(changed) / len(applicable) if applicable else 0.0,
            "truth_hit_repairs": len(repaired),
        }
    for question_type, language, domain in CELLS:
        key = "/".join((question_type, language, domain))
        cell_rows = [row for row in rows if (row["question_type"], row["language"], row["task_domain"]) == (question_type, language, domain)]
        summary["by_cell"][key] = {
            "rows": len(cell_rows),
            "first_truth_hits": sum(bool(row["audit"]["first_truth_hit"]) for row in cell_rows),
            "rows_with_errors": sum(bool(row["audit"].get("errors")) for row in cell_rows),
        }
    for row in rows:
        for error in row.get("audit", {}).get("errors", []):
            summary["errors"].append({"id": row["id"], "error": error})
    return summary


def public_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep truth labels out of the reusable selection manifest."""
    return [{key: value for key, value in row.items() if key != "audit_truth_code"} for row in rows]


def quality_gate(report: dict[str, Any], rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, bool]:
    """Return the pre-SFT hard gate without treating evidence churn as repair."""
    repair_count = sum(
        (not row["audit"]["first_truth_hit"])
        and any(row["audit"]["actions"].get(action, {}).get("truth_hit", False) for action in ("visual_expand", "balanced_compare", "semantic_compare", "name_confirm"))
        for row in rows
    )
    return {
        "complete_cells": not any(report["shortages"].values()),
        "no_row_errors": not report["errors"],
        "all_followup_actions_present": all(set(("visual_expand", "balanced_compare", "semantic_compare", "name_confirm")) <= set(row.get("actions", {})) for row in rows),
        "manifest_has_no_truth": all("audit_truth_code" not in row for row in public_manifest(selected)),
        # This is the substantive HCV feasibility gate.  Different codes alone
        # are not sufficient: a second turn must recover at least one initially
        # absent true class on image-isolated data before we distil it.
        "observed_recall_repair": repair_count > 0,
    }


def main() -> int:
    args = parse_args()
    if args.per_cell < 1 or args.top_k < 1 or args.expand_top_k <= args.top_k:
        raise SystemExit("--per-cell/--top-k must be positive and --expand-top-k must exceed --top-k")
    selected, shortages, isolation_audit = select_rows(read_jsonl(args.source), args.per_cell)
    if any(shortages.values()):
        raise SystemExit(f"insufficient image-isolated rows: {shortages}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "manifest.jsonl", public_manifest(selected))
    rows = []
    for row in selected:
        try:
            rows.append(run_row(row, args))
        except (requests.RequestException, ValueError, KeyError) as exc:
            rows.append({**row, "actions": {}, "audit": {"first_truth_hit": False, "actions": {}, "errors": [f"retrieval_error:{type(exc).__name__}:{exc}"]}})
    write_jsonl(args.output_dir / "audit.jsonl", rows)
    report = {
        "schema_version": "agrinet.hcv-retrieval-preflight/v1",
        "source": str(args.source.relative_to(ROOT)),
        "rag_api": args.rag_api,
        "per_cell": args.per_cell,
        "top_k": args.top_k,
        "expand_top_k": args.expand_top_k,
        "label_policy": "audit-only; labels never form retrieval queries or HTTP requests",
        "isolation": {"policy": "excluded formal/diagnostic/SFT/historical exposed image hashes", **isolation_audit},
        **summarize(rows, shortages),
    }
    report["quality_gate"] = quality_gate(report, rows, selected)
    report["quality_gate"]["passed"] = all(report["quality_gate"].values())
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
