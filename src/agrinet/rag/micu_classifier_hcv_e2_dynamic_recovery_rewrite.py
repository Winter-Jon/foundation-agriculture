"""R0/R1/R2 rewrite recovery for selected E2 dynamic collection winners."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from agrinet.rag.classifier_ledger import BudgetExhausted, DeliveryUnresolved, RequestLedger
from agrinet.rag.micu_classifier_hcv_v2 import sha256
from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget, _invoke_micu, _ledger_contract, run_private_audit
from agrinet.rag.micu_classifier_hcv_e2_dynamic_recovery import _read, _source_rows, _write
from agrinet.rag.micu_classifier_hcv_e2_dynamic_smoke import (
    REWRITE_SCHEMA_V2, REWRITE_SCHEMA_V3, _parse_rewrite, _public_registry_names, _rewrite_prompt,
    render_rewrite_reasoning_v2, rewrite_prompt_v2, rewrite_prompt_v3, validate_rewrite, validate_rewrite_v2,
    validate_rewrite_v3,
)

RECOVERABLE = {"unknown_delivery", "truncated", "invalid_response", "ledger_incomplete", "budget_exhausted"}


def _request_id(directory: Path) -> str | None:
    path = directory / "events.jsonl"
    if not path.is_file():
        return None
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [value.get("request_id") for value in values if value.get("event") == "intent"]
    return str(ids[-1]) if ids and isinstance(ids[-1], str) else None


def _patch_bound_schema(plan: dict[str, Any]) -> tuple[str, str | None]:
    schema, patch_path = str(plan.get("rewrite_schema") or "v1"), plan.get("contract_patch")
    if schema == "v1":
        if patch_path is not None:
            raise ValueError("v1 rewrite manifest must not bind a future contract patch")
        return schema, None
    if schema not in {REWRITE_SCHEMA_V2, REWRITE_SCHEMA_V3} or not isinstance(patch_path, str):
        raise ValueError("rewrite manifest has an unsupported or unbound schema")
    path = Path(patch_path)
    if not path.is_file() or plan.get("contract_patch_sha256") != sha256(path):
        raise ValueError("rewrite manifest contract patch binding is invalid")
    patch = _read(path)
    expected_patch = "0001-rewrite-structure-v2" if schema == REWRITE_SCHEMA_V2 else "0004-image-bound-rewrite-v3"
    if patch.get("patch_id") != expected_patch:
        raise ValueError("rewrite manifest contract patch is not approved for its schema")
    return schema, str(path)


def _collection_winner_rows(collection_summary: Path, *, campaign_id: str, source_sha256: str) -> dict[str, dict[str, Any]]:
    """Index immutable collection rows across available R0/R1/R2 summaries.

    A collection winner may have been closed in a recovery round, while the
    rewrite planner is invoked after R2.  The summaries are append-only and
    colocated, so reading their existing siblings is lineage lookup rather than
    a replay or a mutation.  Keep the explicit supplied summary mandatory for
    small/unit-test layouts where no sibling files exist.
    """
    paths = [collection_summary]
    for round_name in ("r0", "r1", "r2"):
        sibling = collection_summary.parent / f"{round_name}.json"
        if sibling.is_file() and sibling not in paths:
            paths.append(sibling)
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        summary = _read(path)
        # Legacy minimal fixtures predate summary-level source binding.  Real
        # frozen summaries always declare it; when present it is mandatory.
        if (summary.get("campaign_id") != campaign_id
                or (summary.get("source_sha256") is not None and summary.get("source_sha256") != source_sha256)):
            raise ValueError("collection winner summary lineage is inconsistent")
        for row in summary.get("rows", []):
            if not isinstance(row, dict) or not isinstance(row.get("work_id"), str):
                continue
            work_id = str(row["work_id"])
            previous = rows.get(work_id)
            if previous is not None and previous != row:
                raise ValueError("collection winner summary has duplicate work ID")
            rows[work_id] = row
    return rows


def plan_r0(*, campaign_id: str, collection_final: Path, collection_summary: Path,
            source: Path, output: Path, rewrite_schema: str = "v1", contract_patch: Path | None = None) -> dict[str, Any]:
    final, summary = _read(collection_final), _read(collection_summary)
    if final.get("campaign_id") != campaign_id or summary.get("campaign_id") != campaign_id:
        raise ValueError("rewrite inputs must bind the requested campaign")
    source_rows = {str(row["sample_id"]): row for row in _source_rows(source)}
    summary_rows = _collection_winner_rows(collection_summary, campaign_id=campaign_id, source_sha256=sha256(source))
    work = []
    for group in final.get("groups", []):
        if group.get("state") != "selected":
            continue
        winner = summary_rows.get(str(group.get("winner_work_id")))
        sample_id = str(group.get("sample_id") or "")
        parent_path = Path(str((winner or {}).get("parent_path") or ""))
        if winner is None or sample_id not in source_rows or not parent_path.is_file():
            raise ValueError("selected collection winner lacks immutable public parent")
        work.append({"work_id": f"R0:{group['image_group_id']}:rewrite", "round": "R0", "scope": "rewrite",
                     "sample_id": sample_id, "image_group_id": group["image_group_id"],
                     "route": winner["route"], "parent_path": str(parent_path),
                     "collection_winner_request_id": group.get("winner_request_id"),
                     "attempt_ordinal": 0, "predecessor_request_id": None})
    if rewrite_schema == "v1":
        if contract_patch is not None:
            raise ValueError("v1 rewrite plan must not bind a future contract patch")
        patch_fields: dict[str, Any] = {"rewrite_schema": "v1"}
    elif rewrite_schema in {REWRITE_SCHEMA_V2, REWRITE_SCHEMA_V3}:
        if contract_patch is None:
            raise ValueError("rewrite v2 plan requires an immutable contract patch")
        patch_fields = {"rewrite_schema": rewrite_schema, "contract_patch": str(contract_patch),
                        "contract_patch_sha256": sha256(contract_patch)}
        _patch_bound_schema(patch_fields)
    else:
        raise ValueError("rewrite plan schema is unsupported")
    payload = {
        "schema_version": "agrinet.micu-classifier-hcv-e2-dynamic-rewrite-recovery-manifest/v1",
        "campaign_id": campaign_id, "round": "R0", "source": str(source), "source_sha256": sha256(source),
        "collection_final": str(collection_final), "collection_final_sha256": sha256(collection_final),
        "collection_summary": str(collection_summary), "collection_summary_sha256": sha256(collection_summary),
        "work_items": work, "workers": 4, "global_micu_limit": 340,
        "automatic_replay_allowed": False, "training_eligible": False,
        **patch_fields}
    _write(output, payload)
    return payload


def _rewrite_one(item: dict[str, Any], *, source_rows: dict[str, dict[str, Any]],
                 registry_names: list[dict[str, str]], output: Path, campaign_root: Path,
                 teacher_model: str, timeout: int, rewrite_schema: str) -> list[dict[str, Any]]:
    row = source_rows.get(str(item.get("sample_id")))
    parent_path = Path(str(item.get("parent_path") or ""))
    if row is None or not parent_path.is_file():
        raise ValueError("rewrite work lacks frozen source or parent")
    parent = _read(parent_path)
    directory = output / "public" / str(item["image_group_id"])
    ledger = RequestLedger(directory / "ledger", contract=_ledger_contract(teacher_model),
                           image_group_id=row["image_group_id"], view="without_candidates")
    request = ({REWRITE_SCHEMA_V2: rewrite_prompt_v2, REWRITE_SCHEMA_V3: rewrite_prompt_v3}.get(rewrite_schema, _rewrite_prompt))(
        row=row, route=str(item["route"]), parent=parent, registry_names=registry_names)
    request["model"] = teacher_model
    try:
        GlobalMicuBudget(campaign_root / "global_micu_events.jsonl").reserve(item["work_id"] + ":generation")
        raw = ledger.call("generation", "rewrite-1", {"operation": "recovery_rewrite", "sample_id": row["sample_id"], "parent_sha256": sha256(parent_path)}, lambda _summary: _invoke_micu(request, timeout=timeout))
        rewrite = _parse_rewrite(raw)
        errors = ({REWRITE_SCHEMA_V2: validate_rewrite_v2, REWRITE_SCHEMA_V3: validate_rewrite_v3}.get(rewrite_schema, validate_rewrite))(
            row=row, route=str(item["route"]), parent=parent, rewrite=rewrite)
        audit_rewrite = ({"reasoning": render_rewrite_reasoning_v2(rewrite), "final": rewrite["final"]}
                         if rewrite_schema in {REWRITE_SCHEMA_V2, REWRITE_SCHEMA_V3} and not errors else rewrite)
        rewrite_path = directory / "rewrite.json"
        _write(rewrite_path, {"sample_id": row["sample_id"], "route": item["route"], "parent_path": str(parent_path), "rewrite": audit_rewrite, "structured_rewrite": rewrite if rewrite_schema in {REWRITE_SCHEMA_V2, REWRITE_SCHEMA_V3} else None, "rewrite_schema": rewrite_schema, "errors": errors, "status": "closed" if not errors else "quality_rejected", "training_eligible": False})
        outcome = {"work_id": item["work_id"], "planned_work": item, "delivery_status": "delivered", "new_request_id": _request_id(directory / "ledger"), "quality_status": "closed" if not errors else "quality_rejected", "rewrite_path": str(rewrite_path)}
        if errors:
            return [outcome]
        audit_item = {**item, "work_id": item["work_id"].replace(":rewrite", ":rewrite_audit"), "scope": "rewrite_audit", "rewrite_path": str(rewrite_path)}
        try:
            verdict = run_private_audit(source_row=row, parent={**parent, "reasoning_rewrite": audit_rewrite}, output_root=directory, scope="rewrite", timeout=timeout, teacher_model=teacher_model, require_primary_pattern=True, campaign_root=campaign_root, budget_key_prefix=audit_item["work_id"])
            return [outcome, {"work_id": audit_item["work_id"], "planned_work": audit_item, "delivery_status": "delivered", "new_request_id": _request_id(directory / "private" / str(row["sample_id"]) / "rewrite" / "ledger"), "quality_status": verdict["status"], "rewrite_path": str(rewrite_path)}]
        except (DeliveryUnresolved, BudgetExhausted) as exc:
            return [outcome, {"work_id": audit_item["work_id"], "planned_work": audit_item, "delivery_status": "unknown_delivery", "error_type": "budget_exhausted" if isinstance(exc, BudgetExhausted) else type(exc).__name__, "new_request_id": _request_id(directory / "private" / str(row["sample_id"]) / "rewrite" / "ledger") or f"unresolved:{audit_item['work_id']}", "quality_status": None, "rewrite_path": str(rewrite_path)}]
    except (DeliveryUnresolved, BudgetExhausted) as exc:
        return [{"work_id": item["work_id"], "planned_work": item, "delivery_status": "unknown_delivery", "error_type": "budget_exhausted" if isinstance(exc, BudgetExhausted) else type(exc).__name__, "new_request_id": _request_id(directory / "ledger") or f"unresolved:{item['work_id']}", "quality_status": None, "rewrite_path": None}]


def _audit_one(item: dict[str, Any], *, source_rows: dict[str, dict[str, Any]], output: Path,
               campaign_root: Path, teacher_model: str, timeout: int) -> list[dict[str, Any]]:
    """Recover an uncertain private audit without regenerating its closed rewrite."""
    row = source_rows.get(str(item.get("sample_id")))
    rewrite_path = Path(str(item.get("rewrite_path") or ""))
    parent_path = Path(str(item.get("parent_path") or ""))
    if row is None or not rewrite_path.is_file() or not parent_path.is_file():
        raise ValueError("rewrite audit recovery lacks immutable rewrite or parent")
    rewrite_payload, parent = _read(rewrite_path), _read(parent_path)
    rewrite = rewrite_payload.get("rewrite")
    if rewrite_payload.get("status") != "closed" or not isinstance(rewrite, dict):
        raise ValueError("rewrite audit recovery requires a closed rewrite")
    directory = output / "public" / str(item["image_group_id"])
    try:
        verdict = run_private_audit(source_row=row, parent={**parent, "reasoning_rewrite": rewrite},
            output_root=directory, scope="rewrite", timeout=timeout, teacher_model=teacher_model,
            require_primary_pattern=True, campaign_root=campaign_root, budget_key_prefix=item["work_id"])
        return [{"work_id": item["work_id"], "planned_work": item, "delivery_status": "delivered",
                 "new_request_id": _request_id(directory / "private" / str(row["sample_id"]) / "rewrite" / "ledger"),
                 "quality_status": verdict["status"], "rewrite_path": str(rewrite_path)}]
    except (DeliveryUnresolved, BudgetExhausted) as exc:
        return [{"work_id": item["work_id"], "planned_work": item, "delivery_status": "unknown_delivery",
                 "error_type": "budget_exhausted" if isinstance(exc, BudgetExhausted) else type(exc).__name__,
                 "new_request_id": _request_id(directory / "private" / str(row["sample_id"]) / "rewrite" / "ledger") or f"unresolved:{item['work_id']}",
                 "quality_status": None, "rewrite_path": str(rewrite_path)}]


def collect_round(*, manifest: Path, source: Path, public_registry: Path, output: Path,
                  teacher_model: str, timeout: int) -> dict[str, Any]:
    plan = _read(manifest)
    if plan.get("round") not in {"R0", "R1", "R2"} or output.exists():
        raise ValueError("rewrite round output must be new and use R0, R1, or R2 manifest")
    source_rows = {str(row["sample_id"]): row for row in _source_rows(source)}
    registry_sha = str((next(iter(source_rows.values())).get("prediction") or {}).get("registry_sha256") or "")
    registry_names = _public_registry_names(public_registry, expected_sha256=registry_sha)
    items = plan.get("work_items")
    if not isinstance(items, list):
        raise ValueError("rewrite manifest lacks work items")
    rewrite_schema, _ = _patch_bound_schema(plan)
    output.mkdir(parents=True)
    # ``rewrite/manifests/r0.json`` sits one directory below the shared
    # collection campaign root; rewrites must consume the same 340-request
    # ledger rather than opening a phase-local budget.
    campaign_root = manifest.parent.parent.parent
    def run(item: dict[str, Any]) -> list[dict[str, Any]]:
        if item.get("scope") == "rewrite_audit":
            return _audit_one(item, source_rows=source_rows, output=output, campaign_root=campaign_root, teacher_model=teacher_model, timeout=timeout)
        if item.get("scope") != "rewrite":
            raise ValueError("rewrite recovery manifest scope is invalid")
        return _rewrite_one(item, source_rows=source_rows, registry_names=registry_names, output=output, campaign_root=campaign_root, teacher_model=teacher_model, timeout=timeout, rewrite_schema=rewrite_schema)
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = [row for group in pool.map(run, items) for row in group]
    payload = {"schema_version": "agrinet.micu-classifier-hcv-e2-dynamic-rewrite-recovery-outcomes/v1",
               "campaign_id": plan["campaign_id"], "round": plan["round"], "manifest_sha256": sha256(manifest), "rewrite_schema": rewrite_schema,
               "outcomes": outcomes, "automatic_replay_allowed": False, "training_eligible": False}
    _write(output / "outcomes.json", payload)
    return payload


def collect_r0(**kwargs: Any) -> dict[str, Any]:
    return collect_round(**kwargs)


def freeze_summary(*, manifest: Path, outcomes: Path, output: Path) -> dict[str, Any]:
    plan, observed = _read(manifest), _read(outcomes)
    rows = observed.get("outcomes")
    if not isinstance(rows, list):
        raise ValueError("rewrite outcomes must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for item in rows:
        if not isinstance(item, dict) or not isinstance(item.get("work_id"), str):
            raise ValueError("rewrite outcome is invalid")
        if item["work_id"] in indexed:
            raise ValueError("duplicate rewrite outcome")
        if item.get("delivery_status") not in {"delivered", *RECOVERABLE}:
            raise ValueError("rewrite delivery status is invalid")
        indexed[item["work_id"]] = item
    payload = {
        "schema_version": "agrinet.micu-classifier-hcv-e2-dynamic-rewrite-recovery-summary/v1",
        "campaign_id": plan.get("campaign_id"), "round": plan.get("round"),
        "manifest": str(manifest), "manifest_sha256": sha256(manifest),
        "source_sha256": plan.get("source_sha256"), "rows": list(indexed.values()),
        "counts": {"/".join(key): value for key, value in sorted(Counter((str(row["planned_work"].get("scope")), str(row["delivery_status"]), str(row.get("error_type") or "none")) for row in indexed.values()).items())},
        "automatic_replay_allowed": False, "training_eligible": False,
    }
    _write(output, payload)
    return payload


def plan_replenishment(*, summary: Path, next_round: str, output: Path) -> dict[str, Any]:
    if next_round not in {"R1", "R2"}:
        raise ValueError("rewrite replenishment is limited to R1 or R2")
    previous = _read(summary)
    if previous.get("round") != ("R0" if next_round == "R1" else "R1"):
        raise ValueError("rewrite replenishment must follow the immediately preceding summary")
    work = []
    for row in previous.get("rows", []):
        if not isinstance(row, dict) or row.get("delivery_status") == "delivered":
            continue
        item = row.get("planned_work")
        if not isinstance(item, dict):
            raise ValueError("unconfirmed rewrite lacks work lineage")
        ordinal = int(item.get("attempt_ordinal") or 0) + 1
        if ordinal > 2:
            continue
        predecessor = row.get("new_request_id")
        if not isinstance(predecessor, str) or not predecessor:
            raise ValueError("unconfirmed rewrite lacks predecessor request ID")
        work.append({**item, "work_id": item["work_id"].replace(previous["round"], next_round, 1),
                     "round": next_round, "attempt_ordinal": ordinal,
                     "predecessor_request_id": predecessor, "recovery_reason": row["delivery_status"]})
    payload = {
        "schema_version": "agrinet.micu-classifier-hcv-e2-dynamic-rewrite-recovery-manifest/v1",
        "campaign_id": previous.get("campaign_id"), "round": next_round, "source_sha256": previous.get("source_sha256"),
        "previous_summary": str(summary), "previous_summary_sha256": sha256(summary),
        "work_items": work, "workers": 4, "global_micu_limit": 340,
        "automatic_replay_allowed": False, "training_eligible": False,
    }
    _write(output, payload)
    return payload


def final_report(*, collection_final: Path, summaries: list[Path], output: Path) -> dict[str, Any]:
    """Give each selected collection winner one conservative rewrite terminal state."""
    collection = _read(collection_final)
    loaded = [_read(path) for path in summaries]
    if [item.get("round") for item in loaded] != ["R0", "R1", "R2"]:
        raise ValueError("rewrite final report requires frozen R0, R1, and R2 summaries")
    campaign = collection.get("campaign_id")
    if any(item.get("campaign_id") != campaign for item in loaded):
        raise ValueError("collection and rewrite summaries belong to different campaigns")
    selected = {str(row.get("image_group_id")): row for row in collection.get("groups", [])
                if isinstance(row, dict) and row.get("state") == "selected"}
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for summary in loaded:
        for row in summary.get("rows", []):
            if not isinstance(row, dict):
                continue
            work = row.get("planned_work") or {}
            key = (str(work.get("image_group_id") or ""), str(work.get("scope") or ""))
            if key[0] not in selected or key[1] not in {"rewrite", "rewrite_audit"}:
                continue
            prior = latest.get(key)
            if prior is None or int((work.get("attempt_ordinal") or 0)) > int(((prior.get("planned_work") or {}).get("attempt_ordinal") or 0)):
                latest[key] = row
    groups = []
    for group_id, collection_row in sorted(selected.items()):
        rewrite, audit = latest.get((group_id, "rewrite")), latest.get((group_id, "rewrite_audit"))
        rows = [row for row in (rewrite, audit) if row is not None]
        unresolved = [row for row in rows if row.get("delivery_status") != "delivered"]
        tool_failed = [row for row in rows if row.get("quality_status") == "tool_shortfall"]
        if not rewrite:
            state = "delivery_shortfall"
        elif unresolved:
            state = "delivery_shortfall"
        elif tool_failed:
            state = "tool_shortfall"
        elif rewrite.get("quality_status") != "closed":
            state = "quality_rejected"
        elif not audit or audit.get("quality_status") != "accept":
            state = "quality_rejected"
        else:
            state = "accepted"
        groups.append({"image_group_id": group_id, "sample_id": collection_row.get("sample_id"), "state": state,
                       "rewrite_request_id": rewrite.get("new_request_id") if rewrite else None,
                       "audit_request_id": audit.get("new_request_id") if audit else None,
                       "unresolved_scopes": len(unresolved), "attempts": len(rows)})
    counts = Counter(row["state"] for row in groups)
    payload = {"schema_version": "agrinet.micu-classifier-hcv-e2-dynamic-rewrite-recovery-final/v1",
               "campaign_id": campaign, "collection_final": str(collection_final),
               "summaries": [str(path) for path in summaries], "groups": groups, "counts": dict(sorted(counts.items())),
               "automatic_replay_allowed": False, "training_eligible": False, "sft_may_start": False}
    _write(output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    plan = sub.add_parser("plan-r0")
    plan.add_argument("--campaign-id", required=True); plan.add_argument("--collection-final", type=Path, required=True)
    plan.add_argument("--collection-summary", type=Path, required=True); plan.add_argument("--source", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--rewrite-schema", choices=("v1", REWRITE_SCHEMA_V2, REWRITE_SCHEMA_V3), default="v1")
    plan.add_argument("--contract-patch", type=Path)
    collect = sub.add_parser("collect-r0")
    collect.add_argument("--manifest", type=Path, required=True); collect.add_argument("--source", type=Path, required=True)
    collect.add_argument("--public-registry", type=Path, required=True); collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--teacher-model", required=True); collect.add_argument("--timeout", type=int, default=180)
    collect_any = sub.add_parser("collect-round")
    collect_any.add_argument("--manifest", type=Path, required=True); collect_any.add_argument("--source", type=Path, required=True)
    collect_any.add_argument("--public-registry", type=Path, required=True); collect_any.add_argument("--output", type=Path, required=True)
    collect_any.add_argument("--teacher-model", required=True); collect_any.add_argument("--timeout", type=int, default=180)
    freeze = sub.add_parser("freeze-summary")
    freeze.add_argument("--manifest", type=Path, required=True); freeze.add_argument("--outcomes", type=Path, required=True); freeze.add_argument("--output", type=Path, required=True)
    replenish = sub.add_parser("plan-replenishment")
    replenish.add_argument("--summary", type=Path, required=True); replenish.add_argument("--round", choices=("R1", "R2"), required=True); replenish.add_argument("--output", type=Path, required=True)
    final = sub.add_parser("final-report")
    final.add_argument("--collection-final", type=Path, required=True); final.add_argument("--r0-summary", type=Path, required=True)
    final.add_argument("--r1-summary", type=Path, required=True); final.add_argument("--r2-summary", type=Path, required=True); final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.operation == "plan-r0":
        result = plan_r0(campaign_id=args.campaign_id, collection_final=args.collection_final, collection_summary=args.collection_summary, source=args.source, output=args.output, rewrite_schema=args.rewrite_schema, contract_patch=args.contract_patch)
    elif args.operation in {"collect-r0", "collect-round"}:
        result = collect_r0(manifest=args.manifest, source=args.source, public_registry=args.public_registry, output=args.output, teacher_model=args.teacher_model, timeout=args.timeout)
    elif args.operation == "freeze-summary":
        result = freeze_summary(manifest=args.manifest, outcomes=args.outcomes, output=args.output)
    elif args.operation == "plan-replenishment":
        result = plan_replenishment(summary=args.summary, next_round=args.round, output=args.output)
    else:
        result = final_report(collection_final=args.collection_final, summaries=[args.r0_summary, args.r1_summary, args.r2_summary], output=args.output)
    print(json.dumps({key: result.get(key) for key in ("campaign_id", "round")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
