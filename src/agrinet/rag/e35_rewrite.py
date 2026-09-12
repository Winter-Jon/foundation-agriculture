"""Private E3.5 rewrite/rewrite-audit collection for delivered winners."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from agrinet.rag.e35_ledger import DeliveryUnresolved, E35Ledger
from agrinet.rag.e35_private import rewrite_audit_request, rewrite_request, run_rewrite, run_rewrite_audit
from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget

JsonCaller = Callable[[dict[str, Any]], dict[str, Any]]


def rewrite_winner(*, row: dict[str, Any], outcome: dict[str, Any], campaign_root: Path,
                   call: JsonCaller, model: str, intent_limit: int = 8000) -> dict[str, Any]:
    """Perform one immutable rewrite and one isolated rewrite audit."""
    if outcome.get("winner") is not True or outcome.get("delivery_status") != "delivered":
        raise ValueError("rewrite requires a delivered accepted parent winner")
    parent_path = Path(str(outcome.get("parent_path") or ""))
    if not parent_path.is_file(): raise ValueError("rewrite winner lacks immutable parent trajectory")
    trajectory = json.loads(parent_path.read_text(encoding="utf-8"))
    sample_id = str(row.get("sample_id") or "")
    if not sample_id or trajectory.get("route") not in {"direct", "classifier", "rag"}:
        raise ValueError("rewrite parent/source binding is invalid")
    root = Path(campaign_root); directory = root / "rewrite" / sample_id
    ledger = E35Ledger(directory / "ledger", work_id=f"rewrite:{outcome.get('work_id')}", attempt_ordinal=0, intent_limit=intent_limit)
    budget = GlobalMicuBudget(root / "global_micu_intents.jsonl", limit=intent_limit)
    try:
        budget.reserve(f"rewrite:{outcome.get('work_id')}:generation:1")
        rewrite_id, raw_rewrite = ledger.call(kind="rewrite", key="rewrite:1",
            payload={"operation": "rewrite", "sample_id": sample_id, "parent_request_id": outcome.get("request_id")},
            invoke=lambda: call(rewrite_request(trajectory, model=model)))
        rewrite = run_rewrite(trajectory, call=lambda _: raw_rewrite, model=model)
        budget.reserve(f"rewrite:{outcome.get('work_id')}:audit:1")
        audit_id, raw_audit = ledger.call(kind="rewrite_audit", key="rewrite-audit:1",
            payload={"operation": "rewrite_audit", "sample_id": sample_id, "rewrite_request_id": rewrite_id},
            invoke=lambda: call(rewrite_audit_request(rewrite, trajectory, model=model)))
        decision = run_rewrite_audit(rewrite, trajectory, call=lambda _: raw_audit, model=model)
    except DeliveryUnresolved:
        return {"sample_id": sample_id, "winner": False, "delivery_status": "unknown_delivery"}
    target = directory / "rewrite.json"; target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump({"sample_id": sample_id, "route": trajectory["route"], "parent_path": str(parent_path),
                   "rewrite": rewrite, "rewrite_audit": decision, "rewrite_request_id": rewrite_id,
                   "rewrite_audit_request_id": audit_id, "training_eligible": False,
                   "training_authorized": False, "sft_may_start": False}, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
    return {"sample_id": sample_id, "winner": True, "request_id": outcome.get("request_id"),
            "rewrite_request_id": rewrite_id, "trajectory": trajectory, "rewrite": rewrite,
            "rewrite_audit": decision}
