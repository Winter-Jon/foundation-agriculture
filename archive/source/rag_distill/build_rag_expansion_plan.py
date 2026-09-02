#!/usr/bin/env python3
"""Build a teacher-free, fresh-image RAG expansion approval package.

This tool only reads the existing Milvus preflight audit and current strict
corpus.  It never calls the teacher, changes a candidate, or authorizes SFT.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.research.shared.catalog import audited_catalog, candidate_labels, evaluation_images, exposed_images, unknown_delivery_image_hashes
from src.agrinet.data.retrieval_strategies import assign_attempt_strategy

ROOT = Path(__file__).resolve().parents[3]
PRE = ROOT / "outputs/artifacts/datasets/agrinet-rag-recovery-pilot-v5/plan/milvus_preflight.json.jsonl"
SUPPLEMENTAL_PREFLIGHT = ROOT / "outputs/experiments/rag_sft_iteration/approval/rag_expansion_plan_v1/zh_open_pest_catalog_supplement_preflight.jsonl"
STRICT = ROOT / "outputs/experiments/rag_sft_iteration/strict_candidate_view_v1"
OUT = ROOT / "outputs/experiments/rag_sft_iteration/approval/rag_expansion_plan_v1"
TEACHER_MODEL = "gpt-5.6-terra"
SELECTION_SEED = 20260815

# This active builder serves only the approved Stage-A standard milestone.
# Deferred stop-correction work is intentionally outside the current goal and
# must not be made to look like a pending sampling release.
CELLS = {
    f"standard/{q}/{lang}/{domain}": 4
    for q in ("open", "option")
    for lang in ("en", "zh")
    for domain in ("disease", "pest")
}

def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

def image(row: dict[str, Any]) -> str:
    return str(row.get("query_image") or (row.get("metadata") or {}).get("query_image") or (row.get("images") or [""])[0])

def cell(row: dict[str, Any]) -> str:
    return "/".join(str(row.get(k) or (row.get("metadata") or {}).get(k)) for k in ("trajectory_mode", "question_type", "language", "task_domain"))

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")

def main() -> None:
    preflight = read(PRE)
    if SUPPLEMENTAL_PREFLIGHT.exists():
        preflight.extend(read(SUPPLEMENTAL_PREFLIGHT))
    classes = audited_catalog()
    # The preflight audit may contain valid catalog rows not present in the
    # compact historical plans. Use only its class names/domain metadata to
    # complete the public candidate catalog; retrieval results remain separate
    # evidence and are never used as labels.
    for row in preflight:
        code = str(row.get("class_code") or "")
        if code and row.get("class_name") and row.get("class_name_zh") and row.get("task_domain"):
            classes.setdefault(code, {
                "code": code,
                "english_name": str(row["class_name"]),
                "chinese_name": str(row["class_name_zh"]),
                "task_domain": str(row["task_domain"]),
            })
    current = read(STRICT / "rag.current_contract.jsonl") + read(STRICT / "direct.current_contract.jsonl")
    forbidden = set(exposed_images()) | set(evaluation_images()) | {image(row) for row in current}
    unknown_hashes = unknown_delivery_image_hashes()
    existing_cells = Counter(cell(row) for row in read(STRICT / "rag.current_contract.jsonl"))
    deficits = {name: max(0, required - existing_cells.get(name, 0)) for name, required in CELLS.items()}
    fresh = [
        row for row in preflight
        if image(row) not in forbidden
        and row.get("preflight_eligible")
        and hashlib.sha256((ROOT / image(row)).read_bytes()).hexdigest() not in unknown_hashes
    ]
    pools: dict[str, list[dict[str, Any]]] = {}
    for row in fresh:
        pools.setdefault(cell(row), []).append(row)
    selected: list[dict[str, Any]] = []
    cell_report: dict[str, dict[str, Any]] = {}
    used_images: set[str] = set()
    for name in sorted(CELLS):
        need = deficits[name]
        # Quota allocation is deterministic by remaining deficit, but selection
        # within an eligible cell is a reproducible random draw.  This avoids
        # silently treating preflight score/rank as a teacher-quality proxy.
        pool = sorted(pools.get(name, []), key=lambda r: str(r.get("target_id")))
        random.Random(f"{SELECTION_SEED}:{name}").shuffle(pool)
        chosen = []
        if need:
            for row in pool:
                if image(row) in used_images:
                    continue
                chosen.append(row)
                used_images.add(image(row))
                if len(chosen) >= need:
                    break
        for selection_order, row in enumerate(chosen, 1):
            enriched = dict(row)
            # The Milvus preflight audit is retrieval-only and does not carry
            # the public answer contract. Reconstruct that contract here from
            # the audited catalog before any teacher call is authorized.
            target_code = str(enriched.get("class_code") or "")
            if target_code not in classes:
                raise RuntimeError(f"missing audited class metadata for {target_code}")
            if name.split("/")[1] == "option":
                correct_option = str(enriched.get("correct_option") or "")
                if correct_option not in "ABCD":
                    # Deterministic per-cell balancing is assigned in the
                    # selected order; do not infer or rewrite it later.
                    correct_option = "ABCD"[(selection_order - 1) % 4]
                enriched["candidate_labels"] = candidate_labels(
                    classes, target_code, str(enriched.get("task_domain") or ""), correct_option
                )
                enriched["final_label"] = target_code
                enriched["final_label_zh"] = classes[target_code]["chinese_name"]
                enriched["correct_option"] = correct_option
                labels = enriched["candidate_labels"]
                if len(labels) != 4 or labels[ord(correct_option) - ord("A")]["code"] != target_code:
                    raise RuntimeError(f"invalid Option contract for {enriched.get('target_id')}")
            elif name.split("/")[0] == "stop_correction":
                enriched["desired_correction_type"] = "evidence_confirmed"
            enriched["generation_route"] = "blind_evidence"
            enriched["label_visible_to_teacher"] = False
            enriched["approval_only"] = True
            # Target selection order and teacher attempt index are different
            # axes.  Every target emitted here is the first teacher attempt;
            # retries are materialized later by the bulk planner.
            enriched["target_selection_order"] = selection_order
            enriched["candidate_index"] = 1
            enriched = assign_attempt_strategy(enriched, 1)
            selected.append(enriched)
        cell_report[name] = {
            "required": CELLS[name], "existing": existing_cells.get(name, 0),
            "deficit": need, "fresh_eligible_pool": len(pool), "selected": len(chosen),
            "selection_policy": "seeded_uniform_random_within_fresh_eligible_cell",
            "selection_seed": SELECTION_SEED,
            "shortfall": max(0, need - len(chosen)),
        }
    selected.sort(key=lambda r: (cell(r), str(r.get("target_id"))))
    for index, row in enumerate(selected, 1):
        row["approval_sequence"] = index
        row["image_sha256"] = hashlib.sha256((ROOT / image(row)).read_bytes()).hexdigest()
    shards = []
    for start in range(0, len(selected), 8):
        shard = selected[start:start + 8]
        shards.append({"shard": len(shards) + 1, "target_rows": len(shard), "target_ids": [r.get("target_id") for r in shard], "teacher_attempt_cap": min(16, len(shard) * 2), "stop_rule": "stop after each cell reaches four accepted rows or shard cap is exhausted"})
    report = {
        "schema_version": "agrinet.rag-expansion-approval/v1",
        "status": "teacher_free_plan_ready_for_approval",
        "source_preflight": [
            str(PRE.relative_to(ROOT)),
            str(SUPPLEMENTAL_PREFLIGHT.relative_to(ROOT)) if SUPPLEMENTAL_PREFLIGHT.exists() else None,
        ],
        "fresh_exclusion_policy": "exposed plans/candidates + evaluation images + current RAG/Direct images",
        "unknown_delivery_image_hashes_excluded": len(unknown_hashes),
        "preflight_rows": len(preflight), "fresh_preflight_eligible_rows": len(fresh),
        "selected_target_rows": len(selected), "required_target_rows": sum(deficits.values()),
        "within_cell_selection": "seeded_uniform_random_within_fresh_eligible_cell",
        "selection_seed": SELECTION_SEED,
        "teacher_model": TEACHER_MODEL,
        "cell_report": cell_report, "shards": shards,
        "requested_teacher_attempt_cap": 64,
        "route_policy": "Blind-first; teacher sees no private label; Oracle only by explicit follow-up approval if Blind coverage fails",
        "option_policy": "preserve public A/B/C/D mapping from target metadata; no post-hoc letter rewriting",
        "sft_authorized": False, "formal_eval_authorized": False,
        "next_gate": "explicit approval -> local preflight confirmation -> one shard -> strict review -> regenerate gate",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "targets.jsonl", selected)
    standard_targets = selected
    write_jsonl(OUT / "standard_targets.jsonl", standard_targets)
    report["stages"] = {
        "standard_intermediate": {
            "display_name": "standard_milestone",
            "targets": "standard_targets.jsonl",
            "target_rows": len(standard_targets),
            "required_strict_rag_rows": 32,
            "required_direct_rows": 32,
            "sft_authorized": False,
            "formal_eval_authorized": False,
            "next_gate": "stable standard-family Pilot -> adaptive rejection sampling -> immutable 32+32 freeze -> separate milestone SFT approval -> complete phase evaluation -> explicit result decision",
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT / "report.json"), "targets": str(OUT / "targets.jsonl"), "selected": len(selected), "deficits": deficits, "shortfalls": {k: v["shortfall"] for k, v in cell_report.items() if v["shortfall"]}}, ensure_ascii=False))
    raise SystemExit(0 if len(selected) == sum(deficits.values()) and not any(v["shortfall"] for v in cell_report.values()) else 1)

if __name__ == "__main__":
    main()
