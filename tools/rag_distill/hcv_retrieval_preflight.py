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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

from agrinet.data.rebuild_sft import CELLS, image_digest
from tools.rag_distill.build_reconstructive_direct_candidates import isolation_hashes
from tools.rag_distill.catalog_and_isolation import ROOT, read_jsonl, write_jsonl

SOURCE = ROOT / "outputs/vlm_data/disease_pest_large/contrast_samples_vit_base.jsonl"
SEED = "hcv-preflight-v1-20260822"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rag-api", default="http://127.0.0.1:8077")
    parser.add_argument("--per-cell", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def _digest_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(str(row.get("sample_id") or "").encode()).hexdigest()


def select_rows(source: list[dict[str, Any]], per_cell: int, root: Path = ROOT) -> tuple[list[dict[str, Any]], dict[str, int]]:
    forbidden = isolation_hashes(root)
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
    return selected, shortages


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
    if followup is not None:
        anchor, neighbor = followup
        comparison = f"compare {anchor} and {neighbor} symptoms"
        actions["balanced_compare"] = retrieve(args.rag_api, "balanced", image=image, text=comparison, top_k=args.top_k, timeout=args.timeout)
        actions["semantic_compare"] = retrieve(args.rag_api, "semantic", image=None, text=comparison, top_k=args.top_k, timeout=args.timeout)
        actions["name_confirm"] = retrieve(args.rag_api, "name", image=None, text=neighbor, top_k=args.top_k, timeout=args.timeout)
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
    return {**row, "actions": actions, "audit": {"first_truth_hit": truth in first_codes, "actions": actions_audit}}


def summarize(rows: list[dict[str, Any]], shortages: dict[str, int]) -> dict[str, Any]:
    actions = ("balanced_compare", "semantic_compare", "name_confirm")
    summary: dict[str, Any] = {"rows": len(rows), "shortages": shortages, "by_action": {}}
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
    return summary


def main() -> int:
    args = parse_args()
    if args.per_cell < 1 or args.top_k < 1:
        raise SystemExit("--per-cell and --top-k must be positive")
    selected, shortages = select_rows(read_jsonl(args.source), args.per_cell)
    if any(shortages.values()):
        raise SystemExit(f"insufficient image-isolated rows: {shortages}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "manifest.jsonl", selected)
    rows = [run_row(row, args) for row in selected]
    write_jsonl(args.output_dir / "audit.jsonl", rows)
    report = {
        "schema_version": "agrinet.hcv-retrieval-preflight/v1",
        "source": str(args.source.relative_to(ROOT)),
        "rag_api": args.rag_api,
        "per_cell": args.per_cell,
        "top_k": args.top_k,
        "label_policy": "audit-only; labels never form retrieval queries or HTTP requests",
        "isolation": "excluded formal/diagnostic/SFT/historical exposed image hashes via isolation_hashes",
        **summarize(rows, shortages),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
