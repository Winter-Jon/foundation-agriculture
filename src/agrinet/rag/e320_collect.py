"""E3.20 stage-at-a-time four-stage collection controller.

This controller intentionally does not own a provider client.  The live adapter
must inject a stage runner and isolated private auditor.  A work item produces
exactly one public stage trajectory; immutable continuations decide whether the
next request is a quality repair or a semantic escalation.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from agrinet.rag.e320_four_stage_cascade import (
    E320_PROTOCOL, STAGES, safe_private_disposition, validate_e320_trajectory,
    validate_transition,
)
from agrinet.rag.e35_ledger import DeliveryUnresolved
from agrinet.rag.e35_budget import BudgetExhausted

StageRunner = Callable[[dict[str, Any], str, dict[str, Any]], tuple[str, dict[str, Any]]]
AuditRunner = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]]


def _stage(item: dict[str, Any]) -> str:
    route=str(item.get("resume_route") or "direct")
    if route not in STAGES:
        raise ValueError("E3.20 work item has invalid stage")
    progression=item.get("route_progression")
    if progression != list(STAGES):
        raise ValueError("E3.20 work item has invalid frozen progression")
    if int(item.get("quality_attempt_ordinal", 0)) not in {0, 1}:
        raise ValueError("E3.20 work item has invalid quality attempt ordinal")
    if route == "reject" and (not item.get("predecessor_request_id") or not item.get("prior_rag_parent_path")):
        raise ValueError("E3.20 Reject lacks prior RAG lineage")
    return route


def _prior_rag_trajectory(item: dict[str, Any]) -> dict[str, Any] | None:
    path=item.get("prior_rag_parent_path")
    if not isinstance(path, str) or not path:
        return None
    candidate=Path(path)
    if not candidate.is_file():
        raise ValueError("E3.20 Reject prior RAG trajectory is absent")
    value=json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("route") != "rag":
        raise ValueError("E3.20 Reject prior trajectory is not RAG")
    return value


def _write_public(root: Path, work_id: str, stage: str, trajectory: dict[str, Any]) -> Path:
    path=root / "public" / work_id.replace(":", "_") / stage / "trajectory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return path


def run_work_item(*, work_item: dict[str, Any], row: dict[str, Any], output_root: Path,
                  stage_runner: StageRunner, private_audit: AuditRunner) -> dict[str, Any]:
    """Collect and classify one stage without collapsing quality into semantics."""
    if work_item.get("sample_id") != row.get("sample_id"):
        raise ValueError("E3.20 work item/source binding mismatch")
    stage=_stage(work_item)
    prior_rag=_prior_rag_trajectory(work_item) if stage == "reject" else None
    context={"work_id":str(work_item["work_id"]), "stage":stage,
             "attempt_ordinal":int(work_item.get("attempt_ordinal", 0)),
             "quality_attempt_ordinal":int(work_item.get("quality_attempt_ordinal", 0)),
             "prompt_revision":str(work_item.get("prompt_revision", "base")),
             "predecessor_request_id":work_item.get("predecessor_request_id"),
             "prior_rag_trajectory":prior_rag}
    try:
        request_id, trajectory=stage_runner(row, stage, context)
    except DeliveryUnresolved as exc:
        return {"work_id":work_item["work_id"],"sample_id":row["sample_id"],
                "delivery_status":"unknown_delivery","request_id":exc.request_id,
                "winner":False,"final_route":stage,"disposition":"delivery_unknown",
                "quality_attempt_ordinal":context["quality_attempt_ordinal"],"parent_path":None}
    except BudgetExhausted:
        return {"work_id":work_item["work_id"],"sample_id":row["sample_id"],
                "delivery_status":"budget_shortfall","request_id":None,
                "winner":False,"final_route":stage,"disposition":"budget_shortfall",
                "quality_attempt_ordinal":context["quality_attempt_ordinal"],"parent_path":None}
    if not isinstance(request_id, str) or not request_id or not isinstance(trajectory, dict):
        raise ValueError("E3.20 stage runner returned malformed delivery")
    trajectory={**trajectory, "route":stage,
                **({"prior_rag_trajectory":prior_rag} if prior_rag is not None else {})}
    path=_write_public(Path(output_root), str(work_item["work_id"]), stage, trajectory)
    contract_error=None
    try:
        validate_e320_trajectory(row, trajectory)
    except ValueError as exc:
        contract_error=str(exc)
    audit: dict[str, Any]={}
    if contract_error is None:
        try:
            audit=private_audit(row, trajectory, context)
        except DeliveryUnresolved as exc:
            return {"work_id":work_item["work_id"],"sample_id":row["sample_id"],
                    "delivery_status":"unknown_delivery","request_id":exc.request_id,
                    "winner":False,"final_route":stage,"disposition":"delivery_unknown",
                    "quality_attempt_ordinal":context["quality_attempt_ordinal"],"parent_path":str(path)}
        except BudgetExhausted:
            return {"work_id":work_item["work_id"],"sample_id":row["sample_id"],
                    "delivery_status":"budget_shortfall","request_id":None,
                    "winner":False,"final_route":stage,"disposition":"budget_shortfall",
                    "quality_attempt_ordinal":context["quality_attempt_ordinal"],"parent_path":str(path)}
    disposition=safe_private_disposition(audit, contract_error=contract_error)
    # The one allowed same-stage repair has already been consumed.  A second
    # quality failure is a closed, non-winning terminal, not a controller
    # exception that can abort unrelated work in an immutable shard.
    quality_exhausted = (disposition == "quality_reject"
                         and context["quality_attempt_ordinal"] >= 1)
    if not quality_exhausted:
        validate_transition(stage=stage, disposition=disposition,
                            quality_attempt_ordinal=context["quality_attempt_ordinal"])
    return {"work_id":work_item["work_id"],"sample_id":row["sample_id"],
            "delivery_status":"delivered","request_id":request_id,
            "winner":disposition == "semantic_correct","final_route":stage,
            "disposition":disposition,"quality_attempt_ordinal":context["quality_attempt_ordinal"],
            "parent_path":str(path),"contract_error":contract_error,
            "quality_repair_exhausted":quality_exhausted}


def collect_manifest(*, manifest: Path, source: Path, output: Path, output_root: Path,
                     stage_runner: StageRunner, private_audit: AuditRunner) -> dict[str, Any]:
    """Collect exactly the manifest scope and write a bound outcome document."""
    if output.exists():
        raise ValueError("E3.20 outcome destination is immutable")
    plan=json.loads(manifest.read_text(encoding="utf-8"))
    if plan.get("protocol") not in {E320_PROTOCOL, "agrinet.e321-direct-first-hcv-cascade/v1"} or plan.get("schema_version") not in {
        "agrinet.e320-four-stage-cascade-manifest/v1", "agrinet.e321-direct-first-hcv-cascade-manifest/v1"
    }:
        raise ValueError("E3.20 manifest is invalid")
    if any(plan.get(flag) is not False for flag in ("training_eligible", "training_authorized", "sft_may_start")):
        raise ValueError("E3.20 manifest improperly authorizes training")
    rows={str(row.get("sample_id")):row for row in (json.loads(x) for x in source.read_text(encoding="utf-8").splitlines() if x.strip())}
    items=plan.get("work_items") or []
    if not items or len({str(x.get("work_id")) for x in items}) != len(items):
        raise ValueError("E3.20 manifest has invalid work-item scope")
    # Work items are independent at the public/private trajectory level.
    # Ledger and token-budget implementations use file locks, so bounded
    # intra-shard parallelism is safe while still preserving the immutable
    # manifest order in the outcome document.
    workers=int(plan.get("workers", 1))
    if workers < 1 or workers > 4:
        raise ValueError("E3.20 worker count must be within 1..4")
    def run(item: dict[str, Any]) -> dict[str, Any]:
        sample_id=str(item.get("sample_id"))
        if sample_id not in rows: raise ValueError("E3.20 work item source is absent")
        return run_work_item(work_item=item,row=rows[sample_id],output_root=output_root,stage_runner=stage_runner,private_audit=private_audit)
    if workers == 1:
        outcomes=[run(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outcomes=list(executor.map(run,items))
    payload={"schema_version":"agrinet.e320-four-stage-cascade-outcomes/v1","protocol":E320_PROTOCOL,
             "campaign_id":plan.get("campaign_id"),"round":plan.get("round"),
             "manifest_sha256":hashlib.file_digest(manifest.open("rb"),"sha256").hexdigest(),
             "outcomes":outcomes,"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return payload
