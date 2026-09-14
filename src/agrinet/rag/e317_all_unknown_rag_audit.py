"""E3.17: all-simulated-Unknown, three-classifier RAG prospective audit.

This protocol is a new immutable source line.  Its private coverage controls
force each audit row through the cascade; none are projected to the teacher.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.rag.e316_rag_discriminator import PROMPTS, validate_e316_trajectory
from agrinet.rag.e35_classifier_cascade import DELIVERY_FAILURES, recovery_attempt

E317_PROTOCOL = "agrinet.e317-all-unknown-rag-audit/v1"
IDENTITY_KEYS = ("sample_id", "image_sha256", "source_group_id", "near_duplicate_group_id")
CELLS = (("open", "disease"), ("open", "pest"),
         ("option", "disease"), ("option", "pest"))
FOLD_TARGETS = {0: 11, 1: 11, 2: 10}


def _fold(row: dict[str, Any]) -> int:
    value = (row.get("classifier") or {}).get("held_out_fold")
    if value not in FOLD_TARGETS:
        raise ValueError("E3.17 source row has no valid simulated-Unknown classifier fold")
    return int(value)


def _is_disjoint(row: dict[str, Any], blocked: dict[str, set[str]]) -> bool:
    return all(isinstance(row.get(key), str) and row[key] not in blocked[key] for key in IDENTITY_KEYS)


def select_e317_all_unknown(rows: list[dict[str, Any]], *, prior_rows: list[dict[str, Any]],
                            seed: str = "e317-all-unknown-rag-audit-v1") -> list[dict[str, Any]]:
    """Select 32 novel rows: 4 per cell and all three folds in every cell."""
    blocked = {key: {str(row.get(key)) for row in prior_rows if row.get(key) is not None} for key in IDENTITY_KEYS}
    selected: list[dict[str, Any]] = []
    # First take one per fold in each cell (four rows/fold); the remaining
    # 20 positions are allocated deterministically to the frozen 11/11/10 split.
    for kind, domain in CELLS:
        candidates = [row for row in rows if row.get("arm") == "simulated_unknown"
                      and row.get("question_type") == kind and row.get("task_domain") == domain
                      and _is_disjoint(row, blocked)]
        candidates.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest())
        chosen: list[dict[str, Any]] = []
        for fold in FOLD_TARGETS:
            candidate = next((row for row in candidates if _fold(row) == fold), None)
            if candidate is None:
                raise ValueError(f"E3.17 lacks identity-disjoint fold {fold} in {kind}/{domain}")
            chosen.append(candidate)
        selected.extend(chosen)
    counts = Counter(_fold(row) for row in selected)
    while len(selected) < 32:
        # Always fill the least-populated question/domain cell first; ties are
        # fixed by CELLS order.  This preserves 8 rows in every cell.
        kind, domain=min(CELLS, key=lambda cell: (sum(row.get("question_type") == cell[0] and row.get("task_domain") == cell[1] for row in selected), CELLS.index(cell)))
        candidates = [row for row in rows if row.get("arm") == "simulated_unknown"
                      and row.get("question_type") == kind and row.get("task_domain") == domain
                      and _is_disjoint(row, blocked) and row not in selected
                      and counts[_fold(row)] < FOLD_TARGETS[_fold(row)]]
        candidates.sort(key=lambda row: (-(FOLD_TARGETS[_fold(row)] - counts[_fold(row)]),
                                          hashlib.sha256(f"{seed}:fill:{row['sample_id']}".encode()).hexdigest()))
        if not candidates:
            raise ValueError(f"E3.17 cannot meet fold target in {kind}/{domain}")
        selected.append(candidates[0]); counts[_fold(candidates[0])] += 1
    if len(selected) != 32 or counts != Counter(FOLD_TARGETS):
        raise ValueError(f"E3.17 selection balance failed: {dict(counts)}")
    if len({row['sample_id'] for row in selected}) != 32 or any(not _is_disjoint(row, blocked) for row in selected):
        raise ValueError("E3.17 selected identities are not unique/disjoint")
    return selected


def materialize_e317_source(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 32 or Counter(_fold(row) for row in rows) != Counter(FOLD_TARGETS):
        raise ValueError("E3.17 requires 32 rows across classifiers 11/11/10")
    if any(row.get("arm") != "simulated_unknown" for row in rows):
        raise ValueError("E3.17 admits simulated-Unknown rows only")
    cells = Counter((row.get("question_type"), row.get("task_domain")) for row in rows)
    if cells != Counter({cell: 8 for cell in CELLS}):
        raise ValueError("E3.17 requires eight rows per question/domain cell")
    result=[]
    for row in rows:
        private=dict(row.get("private") or {})
        private["e317_route_coverage"] = "rag"
        private["audit_protocol"] = {"rag_witness": True, "purpose": "e317_all_unknown_prospective"}
        result.append({**row, "e39_protocol": E317_PROTOCOL,
                       "teacher_system_prompts": dict(PROMPTS), "private": private})
    return result


def validate_e317_trajectory(row: dict[str, Any], trajectory: dict[str, Any]) -> None:
    try:
        validate_e316_trajectory(row, trajectory)
    except ValueError as exc:
        raise ValueError(str(exc).replace("E3.16", "E3.17")) from exc


def write_e317_manifest(*, source: Path, campaign_id: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("E3.17 manifest is immutable")
    rows=[json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    materialize_e317_source(rows)  # validates the frozen public/private shape
    payload={
        "schema_version": "agrinet.e317-all-unknown-rag-audit-manifest/v1",
        "protocol": E317_PROTOCOL, "campaign_id": campaign_id, "round": "R0",
        "source": str(source), "source_sha256": hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
        "source_rows_expected": 32, "audit_only": True, "all_simulated_unknown": True,
        "classifier_fold_targets": FOLD_TARGETS,
        "work_items": [{"work_id": f"R0:{row['sample_id']}:e317-all-unknown-rag", "round": "R0",
                          "sample_id": row["sample_id"], "image_group_id": row.get("image_group_id", row["image_sha256"]),
                          "attempt_ordinal": 0, "predecessor_request_id": None,
                          "route_progression": ["direct", "classifier", "rag"]} for row in rows],
        "workers": 4, "micu_intent_limit": 8000, "automatic_replay_allowed": False,
        "collection_controls": {"uncached_input_token_cap": 1200000, "transport_image_max_side": 1024,
          "max_public_turns_per_route": 12, "max_rag_searches": 3,
          "reservation_uncached_tokens": {"generation": {"direct": 5000, "classifier": 8000, "rag": 12000}, "private_audit": 18000}},
        "training_eligible": False, "training_authorized": False, "sft_may_start": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def freeze_e317_summary(*, manifest: Path, outcomes: Path, output: Path) -> dict[str, Any]:
    """Freeze a round's exact outcome scope before considering recovery."""
    if output.exists():
        raise ValueError("E3.17 summary is immutable")
    plan=json.loads(manifest.read_text(encoding="utf-8"))
    result=json.loads(outcomes.read_text(encoding="utf-8"))
    actual_sha=hashlib.file_digest(manifest.open("rb"), "sha256").hexdigest()
    if plan.get("schema_version") != "agrinet.e317-all-unknown-rag-audit-manifest/v1" or result.get("manifest_sha256") != actual_sha:
        raise ValueError("E3.17 summary manifest/outcome lineage mismatch")
    items=plan.get("work_items") or []; by_work={str(row.get("work_id")):row for row in result.get("outcomes") or []}
    if len(items) != len(by_work) or {str(item.get("work_id")) for item in items} != set(by_work):
        raise ValueError("E3.17 summary scope mismatch")
    keys=("delivery_status", "request_id", "winner", "final_route", "quality_status", "contract_error", "parent_path", "rag_result")
    rows=[{**{key:item.get(key) for key in ("work_id", "round", "sample_id", "image_group_id", "attempt_ordinal", "predecessor_request_id", "route_progression")},
           **{key:by_work[str(item["work_id"])].get(key) for key in keys}} for item in items]
    payload={"schema_version": "agrinet.e317-all-unknown-rag-audit-summary/v1", "protocol": E317_PROTOCOL,
             "campaign_id": plan.get("campaign_id"), "round": plan.get("round"), "manifest_sha256": actual_sha,
             "source_sha256": plan.get("source_sha256"), "collection_controls": plan.get("collection_controls"),
             "rows": rows, "training_eligible": False, "training_authorized": False, "sft_may_start": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_e317_replenishment_manifest(*, summary: dict[str, Any], next_round: str, output: Path) -> dict[str, Any]:
    prior={"R1": "R0", "R2": "R1"}
    if next_round not in prior or summary.get("round") != prior[next_round]:
        raise ValueError("E3.17 recovery must proceed R0 -> R1 -> R2")
    if output.exists():
        raise ValueError("E3.17 replenishment destination is immutable")
    controls=summary.get("collection_controls")
    if not isinstance(controls, dict) or controls.get("uncached_input_token_cap") != 1200000:
        raise ValueError("E3.17 recovery has no frozen controls")
    work=[]
    for row in summary.get("rows") or []:
        if row.get("delivery_status") not in DELIVERY_FAILURES:
            continue
        rec=recovery_attempt(prior_attempt_ordinal=int(row.get("attempt_ordinal", -1)), status=str(row["delivery_status"]), predecessor_request_id=str(row.get("request_id") or ""))
        if rec.get("new_attempt"):
            work.append({**{key:row[key] for key in ("sample_id", "image_group_id", "route_progression")},
                         "work_id": f"{next_round}:{row['sample_id']}:e317-all-unknown-rag", "round": next_round, **rec})
    payload={"schema_version": "agrinet.e317-all-unknown-rag-audit-manifest/v1", "protocol": E317_PROTOCOL,
             "campaign_id": summary.get("campaign_id"), "round": next_round, "audit_only": True, "all_simulated_unknown": True,
             "work_items": work, "workers": 4, "micu_intent_limit": 8000, "automatic_replay_allowed": False,
             "collection_controls": controls, "training_eligible": False, "training_authorized": False, "sft_may_start": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def e317_final_report(*, source_rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if len(source_rows) != 32 or any(row.get("arm") != "simulated_unknown" for row in source_rows):
        raise ValueError("E3.17 final report requires its all-Unknown source")
    if [item.get("round") for item in summaries] != ["R0", "R1", "R2"]:
        raise ValueError("E3.17 final report requires R0/R1/R2 frozen summaries")
    by_id={str(row.get("sample_id")):row for row in source_rows}
    latest={str(row.get("sample_id")):row for row in summaries[0].get("rows") or []}
    if set(latest) != set(by_id):
        raise ValueError("E3.17 R0 scope does not equal source")
    for summary in summaries[1:]:
        expected={sid for sid,row in latest.items() if row.get("delivery_status") in DELIVERY_FAILURES}
        observed={str(row.get("sample_id")) for row in summary.get("rows") or []}
        if observed != expected:
            raise ValueError("E3.17 recovery summary scope is not exact")
        latest.update({str(row["sample_id"]):row for row in summary.get("rows") or []})
    terminal=Counter(); routes=Counter(); delivery=Counter(); folds=Counter(); cells={f"{kind}/{domain}": Counter() for kind,domain in CELLS}
    for sid, terminal_row in latest.items():
        source=by_id[sid]; route=str(terminal_row.get("final_route") or "none")
        routes[route]+=1; delivery[str(terminal_row.get("delivery_status"))]+=1; folds[str(_fold(source))]+=1
        cell=cells[f"{source['question_type']}/{source['task_domain']}"]
        if terminal_row.get("delivery_status") in {*DELIVERY_FAILURES, "budget_shortfall"}:
            category="delivery_shortfall"
        elif terminal_row.get("winner"):
            category="accepted_rag_refusal" if terminal_row.get("rag_result") == "compliant_refusal" else "accepted"
        else:
            category=str(terminal_row.get("quality_status") or "quality_rejected")
        terminal[category]+=1; cell[category]+=1; cell["closed"]+=1
    return {"schema_version": "agrinet.e317-all-unknown-rag-audit-final-report/v1", "protocol": E317_PROTOCOL,
            "rows_target": 32, "all_simulated_unknown": True, "classifier_fold_counts": dict(folds),
            "final_attempt_routes": dict(routes), "latest_delivery_statuses": dict(delivery), "terminal_counts": dict(terminal), "per_question_domain": {key:dict(value) for key,value in cells.items()},
            "rounds": ["R0", "R1", "R2"], "training_eligible": False, "training_authorized": False, "sft_may_start": False}
