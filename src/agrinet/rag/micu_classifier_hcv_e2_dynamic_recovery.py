"""Experimental round-based delivery recovery for the E2 dynamic smoke.

This module deliberately does not relax or rewrite the historical strict canary
gate. It creates immutable manifests for a separately versioned experiment.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from agrinet.rag.classifier_ledger import BudgetExhausted, DeliveryUnresolved
from agrinet.rag.micu_classifier_hcv_v2 import sha256
from agrinet.rag.micu_classifier_hcv_v2_collect import run_parent, run_private_audit
from agrinet.rag.micu_classifier_hcv_e2_dynamic_smoke import route_contract_errors

RECOVERABLE_DELIVERY = {"unknown_delivery", "truncated", "invalid_response", "ledger_incomplete"}
ROUTE_SLOTS = (("direct", 1), ("direct", 2), ("direct", 3),
               ("classifier", 1), ("classifier", 2), ("rag", 1))


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"immutable destination already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _source_rows(path: Path) -> list[dict[str, Any]]:
    """Read either the formal 32-row source or an explicitly balanced pilot.

    An eight-row source is never a partial formal campaign: it must cover each
    fixed cell once and is labelled as a pilot in every generated manifest.
    """
    value = _read(path)
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) not in {8, 32}:
        raise ValueError("recovery experiment requires an 8-row pilot or frozen 32-image source")
    groups = [str(row.get("image_group_id") or "") for row in rows if isinstance(row, dict)]
    if len(groups) != len(rows) or len(set(groups)) != len(rows) or any(not group for group in groups):
        raise ValueError("source must contain distinct image groups")
    if len(rows) == 8:
        required = {f"{question}-{language}-{domain}" for question in ("open", "option")
                    for language in ("en", "zh") for domain in ("disease", "pest")}
        cells = Counter("-".join(str(row.get(key) or "") for key in ("question_type", "language", "task_domain"))
                        for row in rows)
        if set(cells) != required or any(cells[cell] != 1 for cell in required):
            raise ValueError("8-row pilot source must contain one row in each fixed cell")
    return rows


def _latest_request_id(ledger: Path) -> str | None:
    events = ledger / "events.jsonl"
    if not events.is_file():
        return None
    values = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [value.get("request_id") for value in values if value.get("event") == "intent"]
    return str(ids[-1]) if ids and isinstance(ids[-1], str) else None


def recovery_admission(path: Path, *, endpoint: str, model: str) -> dict[str, Any]:
    """Evaluate the experimental single-round delivery admission.

    The report hash binds this decision to immutable evidence. The historical
    strict gate is recorded, but is never mutated or treated as a pass.
    """
    report = _read(path)
    if report.get("endpoint") != endpoint or report.get("model") != model:
        raise ValueError("canary endpoint or model does not match recovery experiment")
    rounds = report.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 1:
        raise ValueError("recovery admission requires exactly one recorded 10-probe round")
    if not isinstance(rounds[0], dict) or rounds[0].get("requests") != 10:
        raise ValueError("recovery admission requires ten probes")
    detail_path = path.parent / "round-1" / "summary.json"
    detail = _read(detail_path) if detail_path.is_file() else {}
    rows = detail.get("rows")
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("recovery admission requires the raw ten-probe summary")
    statuses: Counter[str] = Counter()
    ready = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("canary probe is invalid")
        if row.get("status") == "delivered" and row.get("valid_response") is True:
            statuses["ready"] += 1
            ready += 1
        elif row.get("status") == "not_completed":
            statuses["unknown_delivery"] += 1
        else:
            statuses["invalid_response"] += 1
    admitted = ready >= 8 and statuses["unknown_delivery"] <= 2 and not statuses["invalid_response"]
    return {
        "schema_version": "agrinet.dynamic-recovery-canary/v1",
        "canary_report": str(path), "canary_report_sha256": sha256(path),
        "endpoint": endpoint, "model": model, "probes": 10,
        "status_counts": dict(sorted(statuses.items())), "ready": ready, "admitted": admitted,
        "historical_strict_gate": report.get("generation_ready_for_dynamic_smoke"),
        "automatic_replay_allowed": False, "training_eligible": False,
    }


def initial_manifest(*, campaign_id: str, source: Path, canary_report: Path, endpoint: str,
                     model: str, output: Path) -> dict[str, Any]:
    admission = recovery_admission(canary_report, endpoint=endpoint, model=model)
    if not admission["admitted"]:
        raise ValueError("recovery admission rejected the canary report")
    rows = _source_rows(source)
    pilot = len(rows) == 8
    # One item per group is executable at the start of R0.  The remaining route
    # slots are a deterministic in-group progression, not pre-authorized calls.
    work = [
        {"work_id": f"R0:{row['image_group_id']}:direct:1:parent", "round": "R0",
         "sample_id": row["sample_id"], "image_group_id": row["image_group_id"],
         "scope": "parent", "route": "direct", "route_attempt": 1,
         "attempt_ordinal": 0, "predecessor_request_id": None, "recovery_reason": None,
         "route_progression": [{"route": route, "route_attempt": ordinal} for route, ordinal in ROUTE_SLOTS]}
        for row in rows
    ]
    payload = {
        "schema_version": "agrinet.micu-classifier-hcv-e2-dynamic-recovery-manifest/v1",
        "campaign_id": campaign_id, "round": "R0", "source": str(source),
        "source_sha256": sha256(source), "source_rows_expected": len(rows), "pilot": pilot,
        "canary_admission": admission, "work_items": work,
        "workers": 4, "global_micu_limit": 340, "automatic_replay_allowed": False,
        "training_eligible": False,
    }
    _write(output, payload)
    return payload


def freeze_round_summary(*, manifest: Path, outcomes: Path, output: Path) -> dict[str, Any]:
    """Freeze one complete round before any subsequent replenishment is planned.

    ``outcomes`` is a runner-produced JSON object with one terminal outcome per
    work id. Delivery problems stop that image group for this round; quality
    outcomes remain terminal and are deliberately not replenishment candidates.
    """
    plan = _read(manifest)
    observed = _read(outcomes).get("outcomes")
    work = plan.get("work_items")
    if not isinstance(work, list) or not isinstance(observed, list):
        raise ValueError("round manifest and outcomes must contain lists")
    expected = {str(item.get("work_id")): item for item in work if isinstance(item, dict)}
    indexed: dict[str, dict[str, Any]] = {}
    for outcome in observed:
        if not isinstance(outcome, dict):
            raise ValueError("round outcome is invalid")
        work_id = str(outcome.get("work_id") or "")
        prior = indexed.get(work_id)
        if prior is None:
            indexed[work_id] = outcome
        elif (prior.get("delivery_status") == "delivered"
              and prior.get("quality_status") == "pending_audit"
              and outcome.get("delivery_status") == "delivered"
              and outcome.get("quality_status") == "tool_shortfall"):
            # Preserve the confirmed parent identity while replacing its
            # provisional audit state with the terminal local-tool failure.
            indexed[work_id] = {**prior, **outcome,
                                "new_request_id": prior.get("new_request_id"),
                                "parent_path": prior.get("parent_path")}
        else:
            raise ValueError("duplicate work outcome is not a valid parent/audit lifecycle pair")
    if len(expected) != len(work) or not set(expected).issubset(indexed):
        raise ValueError("round outcomes must contain all planned work items")
    rows: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, Any]]] = {}
    for work_id, outcome in indexed.items():
        item = expected.get(work_id, outcome.get("planned_work"))
        if not isinstance(item, dict):
            raise ValueError("outcome lacks immutable work lineage")
        delivery = str(outcome.get("delivery_status") or "")
        error_type = str(outcome.get("error_type") or "")
        if delivery not in {"delivered", *RECOVERABLE_DELIVERY}:
            raise ValueError("outcome delivery_status is invalid")
        merged = {**item, "delivery_status": delivery, "error_type": error_type or None,
                  "new_request_id": outcome.get("new_request_id"),
                  "payload_sha256": outcome.get("payload_sha256"),
                  "latency_seconds": outcome.get("latency_seconds"),
                  "quality_status": outcome.get("quality_status"),
                  "route_contract_status": outcome.get("route_contract_status"),
                  "selected_winner": bool(outcome.get("selected_winner", False))}
        if delivery != "delivered" and not merged["error_type"]:
            raise ValueError("unconfirmed delivery requires an error type")
        rows.append(merged)
        by_group.setdefault(str(item["image_group_id"]), []).append(merged)
    for group_rows in by_group.values():
        pending = any(row["delivery_status"] != "delivered" for row in group_rows)
        for row in group_rows:
            row["group_delivery_pending"] = pending
            row["selected_winner"] = row["selected_winner"] and not pending
    counts = Counter((row["scope"], row["delivery_status"], row.get("error_type") or "none") for row in rows)
    payload = {
        "schema_version": "agrinet.micu-classifier-hcv-e2-dynamic-recovery-summary/v1",
        "campaign_id": plan.get("campaign_id"), "round": plan.get("round"),
        "manifest": str(manifest), "manifest_sha256": sha256(manifest),
        "source_sha256": plan.get("source_sha256"), "source_rows_expected": plan.get("source_rows_expected"),
        "pilot": bool(plan.get("pilot")), "canary_admission_sha256": (plan.get("canary_admission") or {}).get("canary_report_sha256"),
        "rows": rows, "counts": {"/".join(key): value for key, value in sorted(counts.items())},
        "automatic_replay_allowed": False, "training_eligible": False,
    }
    _write(output, payload)
    return payload


def replenishment_manifest(*, summary: Path, next_round: str, output: Path) -> dict[str, Any]:
    """Plan R1/R2 only after a frozen previous-round summary.

    A logical scope receives at most two new attempts. Each item has fresh
    output lineage and refers to, rather than reuses, the predecessor request.
    """
    if next_round not in {"R1", "R2"}:
        raise ValueError("replenishment is limited to R1 or R2")
    previous = _read(summary)
    expected_previous = "R0" if next_round == "R1" else "R1"
    if previous.get("round") != expected_previous:
        raise ValueError("replenishment must follow the immediately preceding frozen round")
    rows = previous.get("rows")
    if not isinstance(rows, list):
        raise ValueError("summary lacks rows")
    work = []
    for row in rows:
        if not isinstance(row, dict) or row.get("delivery_status") == "delivered":
            continue
        if str(row.get("delivery_status")) not in RECOVERABLE_DELIVERY:
            continue
        ordinal = int(row.get("attempt_ordinal", -1)) + 1
        if ordinal > 2:
            continue
        predecessor = row.get("new_request_id")
        if not isinstance(predecessor, str) or not predecessor:
            raise ValueError("unconfirmed delivery must retain predecessor request ID")
        work.append({
            **{key: row[key] for key in ("sample_id", "image_group_id", "scope", "route", "route_attempt")},
            "work_id": f"{next_round}:{row['image_group_id']}:{row['route']}:{row['route_attempt']}:{row['scope']}",
            "round": next_round, "attempt_ordinal": ordinal, "predecessor_request_id": predecessor,
            "predecessor_payload_sha256": row.get("payload_sha256"),
            "parent_path": row.get("parent_path"),
            "recovery_reason": row.get("delivery_status"),
        })
    payload = {
        "schema_version": "agrinet.micu-classifier-hcv-e2-dynamic-recovery-manifest/v1",
        "campaign_id": previous.get("campaign_id"), "round": next_round,
        "source_sha256": previous.get("source_sha256"), "source_rows_expected": previous.get("source_rows_expected"),
        "pilot": bool(previous.get("pilot")), "previous_summary": str(summary),
        "previous_summary_sha256": sha256(summary), "work_items": work, "workers": 4,
        "global_micu_limit": 340, "automatic_replay_allowed": False, "training_eligible": False,
    }
    _write(output, payload)
    return payload


def collect_round(*, manifest: Path, source: Path, output: Path, rag_endpoint: str,
                  teacher_model: str, timeout: int, max_tokens: int) -> dict[str, Any]:
    """Execute one immutable round with four independent image-group workers.

    This is intentionally not invoked by the R0 planning experiment. A delivery
    uncertainty ends the group for the round; it never advances to another
    route in the same round. The returned outcome file is the only input to a
    later frozen summary/replenishment plan.
    """
    plan = _read(manifest)
    if output.exists():
        raise ValueError("round output already exists")
    source_rows = _source_rows(source)
    if plan.get("source_sha256") != sha256(source) or plan.get("source_rows_expected") != len(source_rows):
        raise ValueError("round manifest does not bind the supplied frozen source")
    if bool(plan.get("pilot")) != (len(source_rows) == 8):
        raise ValueError("round manifest pilot designation does not match the frozen source")
    source_by_id = {str(row["sample_id"]): row for row in source_rows}
    items = plan.get("work_items")
    if not isinstance(items, list):
        raise ValueError("manifest lacks work items")
    output.mkdir(parents=True)
    campaign_root = manifest.parent.parent

    def run_one(initial: dict[str, Any]) -> list[dict[str, Any]]:
        row = source_by_id.get(str(initial.get("sample_id")))
        if row is None:
            raise ValueError("manifest sample is absent from frozen source")
        if initial.get("scope") == "parent_audit":
            parent_path = Path(str(initial.get("parent_path") or ""))
            if not parent_path.is_file():
                raise ValueError("audit recovery requires its closed predecessor parent")
            work = dict(initial)
            try:
                parent = _read(parent_path)
                audit = run_private_audit(source_row=row, parent=parent, output_root=output / "attempts" / str(initial["image_group_id"]), scope="parent", timeout=timeout, teacher_model=teacher_model, require_primary_pattern=True, campaign_root=campaign_root)
                contract = route_contract_errors(parent, str(initial["route"]), audit_status=audit["status"])
                return [{"work_id": work["work_id"], "planned_work": work, "delivery_status": "delivered",
                         "new_request_id": _latest_request_id(output / "attempts" / str(initial["image_group_id"]) / "private" / str(row["sample_id"]) / "parent" / "ledger"),
                         "quality_status": audit["status"], "route_contract_status": "pass" if not contract else "reject",
                         "route_contract_errors": contract, "selected_winner": audit["status"] == "accept" and not contract, "parent_path": str(parent_path)}]
            except DeliveryUnresolved as exc:
                return [{"work_id": work["work_id"], "planned_work": work, "delivery_status": "unknown_delivery", "error_type": type(exc).__name__, "new_request_id": _latest_request_id(output / "attempts" / str(initial["image_group_id"]) / "private" / str(row["sample_id"]) / "parent" / "ledger") or f"unresolved:{work['work_id']}", "quality_status": None, "route_contract_status": None, "selected_winner": False, "parent_path": str(parent_path)}]
        progression = (initial.get("route_progression") if plan.get("round") == "R0"
                       else [{"route": initial["route"], "route_attempt": initial["route_attempt"]}])
        outcomes = []
        for route_slot in progression:
            route, ordinal = str(route_slot["route"]), int(route_slot["route_attempt"])
            work = {**initial, "route": route, "route_attempt": ordinal,
                    "work_id": f"{plan['round']}:{initial['image_group_id']}:{route}:{ordinal}:parent"}
            location = output / "attempts" / str(initial["image_group_id"]) / route / str(ordinal)
            tools = set() if route == "direct" else ({"agrinet_classifier_predict", "agrinet_classifier_expand"} if route == "classifier" else {"agrinet_classifier_predict", "agrinet_classifier_expand", "agrinet_rag_search"})
            try:
                parent = run_parent(source_row=row, output_root=location, rag_endpoint=rag_endpoint, max_tokens=max_tokens, timeout=timeout, teacher_model=teacher_model, temperature=0.5 if route == "direct" else 0.2, allowed_tools=tools, campaign_root=campaign_root, budget_key_prefix=work["work_id"])
                parent_path = location / "public" / str(row["sample_id"]) / "trajectory.json"
                outcomes.append({"work_id": work["work_id"], "planned_work": work, "delivery_status": "delivered", "new_request_id": _latest_request_id(location / "public" / str(row["sample_id"]) / "ledger"), "quality_status": "pending_audit", "route_contract_status": "unchecked", "selected_winner": False, "parent_path": str(parent_path)})
                audit_work = {**work, "work_id": work["work_id"].replace(":parent", ":parent_audit"), "scope": "parent_audit", "parent_path": str(parent_path)}
                try:
                    audit = run_private_audit(source_row=row, parent=parent, output_root=location, scope="parent", timeout=timeout, teacher_model=teacher_model, require_primary_pattern=True, campaign_root=campaign_root, budget_key_prefix=audit_work["work_id"])
                    contract = route_contract_errors(parent, route, audit_status=audit["status"])
                    outcomes.append({"work_id": audit_work["work_id"], "planned_work": audit_work, "delivery_status": "delivered", "new_request_id": _latest_request_id(location / "private" / str(row["sample_id"]) / "parent" / "ledger"), "quality_status": audit["status"], "route_contract_status": "pass" if not contract else "reject", "route_contract_errors": contract, "selected_winner": audit["status"] == "accept" and not contract, "parent_path": str(parent_path)})
                except DeliveryUnresolved as exc:
                    outcomes.append({"work_id": audit_work["work_id"], "planned_work": audit_work, "delivery_status": "unknown_delivery", "error_type": type(exc).__name__, "new_request_id": _latest_request_id(location / "private" / str(row["sample_id"]) / "parent" / "ledger") or f"unresolved:{audit_work['work_id']}", "quality_status": None, "route_contract_status": None, "selected_winner": False, "parent_path": str(parent_path)})
                    break
                if audit["status"] == "accept":
                    break
            except (DeliveryUnresolved, BudgetExhausted) as exc:
                outcomes.append({"work_id": work["work_id"], "planned_work": work, "delivery_status": "unknown_delivery",
                                 "error_type": "budget_exhausted" if isinstance(exc, BudgetExhausted) else type(exc).__name__, "new_request_id": _latest_request_id(location / "public" / str(row["sample_id"]) / "ledger") or f"unresolved:{work['work_id']}", "quality_status": None, "route_contract_status": None, "selected_winner": False})
                break
            except (RuntimeError, ValueError) as exc:
                if outcomes and outcomes[-1].get("work_id") == work["work_id"]:
                    outcomes[-1].update({"quality_status": "tool_shortfall", "error_type": type(exc).__name__,
                                         "route_contract_status": None, "selected_winner": False})
                else:
                    outcomes.append({"work_id": work["work_id"], "planned_work": work, "delivery_status": "delivered",
                                     "new_request_id": _latest_request_id(location / "public" / str(row["sample_id"]) / "ledger"),
                                     "quality_status": "tool_shortfall", "error_type": type(exc).__name__,
                                     "route_contract_status": None, "selected_winner": False})
                break
        return outcomes

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = [row for group in pool.map(run_one, items) for row in group]
    payload = {"schema_version": "agrinet.micu-classifier-hcv-e2-dynamic-recovery-outcomes/v1",
               "campaign_id": plan.get("campaign_id"), "round": plan.get("round"), "manifest_sha256": sha256(manifest),
               "outcomes": outcomes, "automatic_replay_allowed": False, "training_eligible": False}
    _write(output / "outcomes.json", payload)
    return payload


def final_report(*, summaries: list[Path], output: Path) -> dict[str, Any]:
    """Report exactly one terminal state per image group after R0--R2.

    Conversion must consume only ``selected`` groups.  A quality rejection is
    distinct from provider delivery shortfall and from local tool failure.
    """
    loaded = [_read(path) for path in summaries]
    if [item.get("round") for item in loaded] != ["R0", "R1", "R2"]:
        raise ValueError("final report requires frozen R0, R1, and R2 summaries")
    campaign = loaded[0].get("campaign_id")
    if any(item.get("campaign_id") != campaign for item in loaded):
        raise ValueError("summaries belong to different campaigns")
    all_rows = [row for item in loaded for row in item.get("rows", []) if isinstance(row, dict)]
    latest: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in all_rows:
        key = (str(row.get("image_group_id") or ""), str(row.get("route") or ""),
               int(row.get("route_attempt") or 0), str(row.get("scope") or ""))
        previous = latest.get(key)
        if previous is None or int(row.get("attempt_ordinal") or 0) > int(previous.get("attempt_ordinal") or 0):
            latest[key] = row
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in latest.values():
        groups.setdefault(str(row.get("image_group_id") or ""), []).append(row)
    terminal = []
    for group, rows in sorted(groups.items()):
        if not group:
            raise ValueError("summary row lacks image group")
        winners = [row for row in rows if row.get("selected_winner") is True]
        unresolved = [row for row in rows if row.get("delivery_status") != "delivered"]
        tool_failed = [row for row in rows if row.get("quality_status") == "tool_shortfall"]
        if len(winners) > 1:
            raise ValueError("image group has more than one selected winner")
        if winners and not unresolved:
            state, winner = "selected", winners[0]
        elif unresolved:
            state, winner = "delivery_shortfall", None
        elif tool_failed:
            state, winner = "tool_shortfall", None
        else:
            state, winner = "quality_rejected", None
        terminal.append({
            "image_group_id": group, "sample_id": rows[0].get("sample_id"), "state": state,
            "winner_work_id": winner.get("work_id") if winner else None,
            "winner_request_id": winner.get("new_request_id") if winner else None,
            "unresolved_scopes": len(unresolved), "attempts": len(rows),
        })
    counts = Counter(row["state"] for row in terminal)
    payload = {
        "schema_version": "agrinet.micu-classifier-hcv-e2-dynamic-recovery-final/v1",
        "campaign_id": campaign, "summaries": [str(path) for path in summaries],
        "groups": terminal, "counts": dict(sorted(counts.items())),
        "automatic_replay_allowed": False, "training_eligible": False, "sft_may_start": False,
    }
    _write(output, payload)
    return payload


def conversion_gate(*, collection_final: Path, rewrite_final: Path, output: Path) -> dict[str, Any]:
    """Admit only fully audited recovery winners to a conversion candidate list."""
    collection = _read(collection_final)
    rewrite = _read(rewrite_final)
    if collection.get("campaign_id") != rewrite.get("campaign_id"):
        raise ValueError("collection and rewrite reports belong to different campaigns")
    collection_groups = collection.get("groups")
    rewrite_groups = rewrite.get("groups")
    if not isinstance(collection_groups, list) or not isinstance(rewrite_groups, list):
        raise ValueError("final reports lack image-group rows")
    rewrites = {str(row.get("image_group_id") or ""): row for row in rewrite_groups if isinstance(row, dict)}
    if len(rewrites) != len(rewrite_groups):
        raise ValueError("rewrite final has duplicate or missing image groups")
    accepted, excluded = [], []
    for row in collection_groups:
        if not isinstance(row, dict):
            raise ValueError("collection final row is invalid")
        group = str(row.get("image_group_id") or "")
        review = rewrites.get(group)
        if row.get("state") == "selected" and review and review.get("state") == "accepted":
            accepted.append({"image_group_id": group, "sample_id": row.get("sample_id"),
                             "collection_winner_request_id": row.get("winner_request_id"),
                             "rewrite_request_id": review.get("rewrite_request_id"),
                             "rewrite_audit_request_id": review.get("audit_request_id")})
        else:
            excluded.append({"image_group_id": group, "sample_id": row.get("sample_id"),
                             "collection_state": row.get("state"),
                             "rewrite_state": review.get("state") if review else "missing"})
    payload = {
        "schema_version": "agrinet.micu-classifier-hcv-e2-dynamic-recovery-conversion-gate/v1",
        "campaign_id": collection.get("campaign_id"), "collection_final": str(collection_final),
        "rewrite_final": str(rewrite_final), "accepted": accepted, "excluded": excluded,
        "automatic_replay_allowed": False, "training_eligible": False, "sft_may_start": False,
    }
    _write(output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    initial = sub.add_parser("plan-r0")
    initial.add_argument("--campaign-id", required=True); initial.add_argument("--source", type=Path, required=True)
    initial.add_argument("--canary-report", type=Path, required=True); initial.add_argument("--endpoint", required=True)
    initial.add_argument("--model", required=True); initial.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("freeze-summary")
    freeze.add_argument("--manifest", type=Path, required=True); freeze.add_argument("--outcomes", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    collect = sub.add_parser("collect-round")
    collect.add_argument("--manifest", type=Path, required=True); collect.add_argument("--source", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True); collect.add_argument("--rag-endpoint", required=True)
    collect.add_argument("--teacher-model", required=True); collect.add_argument("--timeout", type=int, default=180)
    collect.add_argument("--max-tokens", type=int, default=8192)
    replenish = sub.add_parser("plan-replenishment")
    replenish.add_argument("--summary", type=Path, required=True); replenish.add_argument("--round", choices=("R1", "R2"), required=True)
    replenish.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("final-report")
    report.add_argument("--r0-summary", type=Path, required=True); report.add_argument("--r1-summary", type=Path, required=True)
    report.add_argument("--r2-summary", type=Path, required=True); report.add_argument("--output", type=Path, required=True)
    convert_gate = sub.add_parser("conversion-gate")
    convert_gate.add_argument("--collection-final", type=Path, required=True)
    convert_gate.add_argument("--rewrite-final", type=Path, required=True)
    convert_gate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.operation == "plan-r0":
        result = initial_manifest(campaign_id=args.campaign_id, source=args.source, canary_report=args.canary_report, endpoint=args.endpoint, model=args.model, output=args.output)
    elif args.operation == "freeze-summary":
        result = freeze_round_summary(manifest=args.manifest, outcomes=args.outcomes, output=args.output)
    elif args.operation == "collect-round":
        result = collect_round(manifest=args.manifest, source=args.source, output=args.output, rag_endpoint=args.rag_endpoint, teacher_model=args.teacher_model, timeout=args.timeout, max_tokens=args.max_tokens)
    elif args.operation == "plan-replenishment":
        result = replenishment_manifest(summary=args.summary, next_round=args.round, output=args.output)
    elif args.operation == "conversion-gate":
        result = conversion_gate(collection_final=args.collection_final, rewrite_final=args.rewrite_final, output=args.output)
    else:
        result = final_report(summaries=[args.r0_summary, args.r1_summary, args.r2_summary], output=args.output)
    print(json.dumps({key: result.get(key) for key in ("campaign_id", "round", "counts")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
