#!/usr/bin/env python3
"""Build a teacher-free, fresh-image RAG expansion approval package.

This tool only reads the existing Milvus preflight audit and current strict
corpus.  It never calls the teacher, changes a candidate, or authorizes SFT.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.rag_distill.build_round111_option_coverage import exposed_images, evaluation_images
from tools.rag_distill.build_round098_option_coverage import audited_catalog, candidate_labels
from src.agrinet.data.retrieval_strategies import assign_attempt_strategy

ROOT = Path(__file__).resolve().parents[2]
PRE = ROOT / "outputs/artifacts/datasets/agrinet-rag-recovery-pilot-v5/plan/milvus_preflight.json.jsonl"
STRICT = ROOT / "outputs/experiments/rag_sft_iteration/strict_candidate_view_v1"
OUT = ROOT / "outputs/experiments/rag_sft_iteration/approval/rag_expansion_plan_v1"
TEACHER_MODEL = "gpt-5.6-terra"

CELLS = {
    **{f"standard/{q}/{lang}/{domain}": 4 for q in ("open", "option") for lang in ("en", "zh") for domain in ("disease", "pest")},
    **{f"stop_correction/open/{lang}/{domain}": 4 for lang in ("en", "zh") for domain in ("disease", "pest")},
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
    existing_cells = Counter(cell(row) for row in read(STRICT / "rag.current_contract.jsonl"))
    deficits = {name: max(0, required - existing_cells.get(name, 0)) for name, required in CELLS.items()}
    fresh = [row for row in preflight if image(row) not in forbidden and row.get("preflight_eligible")]
    pools: dict[str, list[dict[str, Any]]] = {}
    for row in fresh:
        pools.setdefault(cell(row), []).append(row)
    selected: list[dict[str, Any]] = []
    cell_report: dict[str, dict[str, Any]] = {}
    used_images: set[str] = set()
    for name in sorted(CELLS):
        need = deficits[name]
        pool = sorted(pools.get(name, []), key=lambda r: (int(r.get("preflight_target_rank") or 99), -float(r.get("preflight_target_score") or 0), str(r.get("target_id"))))
        chosen = []
        if need:
            for row in pool:
                if image(row) in used_images:
                    continue
                chosen.append(row)
                used_images.add(image(row))
                if len(chosen) >= need:
                    break
        for index, row in enumerate(chosen, 1):
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
                    correct_option = "ABCD"[(index - 1) % 4]
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
            enriched["candidate_index"] = index
            enriched = assign_attempt_strategy(enriched, index)
            selected.append(enriched)
        cell_report[name] = {
            "required": CELLS[name], "existing": existing_cells.get(name, 0),
            "deficit": need, "fresh_eligible_pool": len(pool), "selected": len(chosen),
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
        "source_preflight": str(PRE.relative_to(ROOT)),
        "fresh_exclusion_policy": "exposed plans/candidates + evaluation images + current RAG/Direct images",
        "preflight_rows": len(preflight), "fresh_preflight_eligible_rows": len(fresh),
        "selected_target_rows": len(selected), "required_target_rows": sum(deficits.values()),
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
    standard_targets = [row for row in selected if str(row.get("trajectory_mode")) == "standard"]
    deferred_stop_correction_targets = [row for row in selected if str(row.get("trajectory_mode")) == "stop_correction"]
    write_jsonl(OUT / "standard_targets.jsonl", standard_targets)
    write_jsonl(OUT / "deferred_stop_correction_targets.jsonl", deferred_stop_correction_targets)
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
        "full_milestone": {
            "targets": "targets.jsonl",
            "deferred_stop_correction_targets": "deferred_stop_correction_targets.jsonl",
            "deferred_target_rows": len(deferred_stop_correction_targets),
            "required_strict_rag_rows": 48,
            "required_direct_rows": 32,
            "sft_authorized": False,
            "formal_eval_authorized": False,
            "next_gate": "repair and validate stop-correction -> fill remaining 16 rows -> immutable 48+32 freeze -> separate milestone SFT approval",
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT / "report.json"), "targets": str(OUT / "targets.jsonl"), "selected": len(selected), "deficits": deficits, "shortfalls": {k: v["shortfall"] for k, v in cell_report.items() if v["shortfall"]}}, ensure_ascii=False))
    raise SystemExit(0 if len(selected) == sum(deficits.values()) and not any(v["shortfall"] for v in cell_report.values()) else 1)

if __name__ == "__main__":
    main()
