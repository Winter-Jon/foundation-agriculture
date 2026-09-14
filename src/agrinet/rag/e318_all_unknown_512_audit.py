"""E3.18: a new 512px all-simulated-Unknown full-32 budget audit.

This is deliberately source- and lineage-disjoint from terminal E3.17.  It
uses the same discriminative RAG/Hermes contract to measure the real cost of
completing all 32 forced cascades at a lower image transport resolution.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.rag.e316_rag_discriminator import PROMPTS
from agrinet.rag.e317_all_unknown_rag_audit import (
    FOLD_TARGETS, _fold, materialize_e317_source, select_e317_all_unknown,
    validate_e317_trajectory,
)
from agrinet.rag.e35_classifier_cascade import DELIVERY_FAILURES, recovery_attempt

E318_PROTOCOL = "agrinet.e318-all-unknown-512-rag-audit/v1"
E318_TOKEN_CAP = 8_000_000


def select_e318_all_unknown(rows: list[dict[str, Any]], *, prior_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return select_e317_all_unknown(rows, prior_rows=prior_rows, seed="e318-all-unknown-512-rag-audit-v1")


def materialize_e318_source(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate E3.17's balance, then substitute E3.18-only private control."""
    checked=materialize_e317_source(rows)
    result=[]
    for row in checked:
        private=dict(row.get("private") or {})
        private.pop("e317_route_coverage", None)
        private["e318_route_coverage"] = "rag"
        private["audit_protocol"] = {"rag_witness": True, "purpose": "e318_512_full_budget_calibration"}
        result.append({**row, "e39_protocol": E318_PROTOCOL, "teacher_system_prompts": dict(PROMPTS), "private": private})
    return result


def validate_e318_trajectory(row: dict[str, Any], trajectory: dict[str, Any]) -> None:
    try:
        validate_e317_trajectory(row, trajectory)
    except ValueError as exc:
        raise ValueError(str(exc).replace("E3.17", "E3.18")) from exc


def write_e318_manifest(*, source: Path, campaign_id: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("E3.18 manifest is immutable")
    rows=[json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 32 or any(row.get("e39_protocol") != E318_PROTOCOL for row in rows):
        raise ValueError("E3.18 source binding invalid")
    payload={
        "schema_version": "agrinet.e318-all-unknown-512-rag-audit-manifest/v1",
        "protocol": E318_PROTOCOL, "campaign_id": campaign_id, "round": "R0",
        "source": str(source), "source_sha256": hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
        "source_rows_expected": 32, "audit_only": True, "all_simulated_unknown": True,
        "classifier_fold_targets": FOLD_TARGETS, "transport_resolution_calibration": True,
        "work_items": [{"work_id": f"R0:{row['sample_id']}:e318-all-unknown-512-rag", "round": "R0",
                          "sample_id": row["sample_id"], "image_group_id": row.get("image_group_id", row["image_sha256"]),
                          "attempt_ordinal": 0, "predecessor_request_id": None,
                          "route_progression": ["direct", "classifier", "rag"]} for row in rows],
        "workers": 4, "micu_intent_limit": 8000, "automatic_replay_allowed": False,
        "collection_controls": {"uncached_input_token_cap": E318_TOKEN_CAP, "transport_image_max_side": 512,
          "max_public_turns_per_route": 12, "max_rag_searches": 3,
          "reservation_uncached_tokens": {"generation": {"direct": 5000, "classifier": 8000, "rag": 12000}, "private_audit": 18000}},
        "training_eligible": False, "training_authorized": False, "sft_may_start": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def freeze_e318_summary(*, manifest: Path, outcomes: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("E3.18 summary is immutable")
    plan=json.loads(manifest.read_text(encoding="utf-8")); result=json.loads(outcomes.read_text(encoding="utf-8"))
    actual=hashlib.file_digest(manifest.open("rb"), "sha256").hexdigest()
    if plan.get("schema_version") != "agrinet.e318-all-unknown-512-rag-audit-manifest/v1" or result.get("manifest_sha256") != actual:
        raise ValueError("E3.18 summary manifest/outcome lineage mismatch")
    items=plan.get("work_items") or []; outcomes_by_work={str(row.get("work_id")):row for row in result.get("outcomes") or []}
    if len(items) != len(outcomes_by_work) or {str(item.get("work_id")) for item in items} != set(outcomes_by_work):
        raise ValueError("E3.18 summary scope mismatch")
    keys=("delivery_status", "request_id", "winner", "final_route", "quality_status", "contract_error", "parent_path", "rag_result")
    rows=[{**{key:item.get(key) for key in ("work_id", "round", "sample_id", "image_group_id", "attempt_ordinal", "predecessor_request_id", "route_progression")},
           **{key:outcomes_by_work[str(item["work_id"])].get(key) for key in keys}} for item in items]
    payload={"schema_version": "agrinet.e318-all-unknown-512-rag-audit-summary/v1", "protocol": E318_PROTOCOL,
             "campaign_id":plan.get("campaign_id"), "round":plan.get("round"), "manifest_sha256":actual,
             "source_sha256":plan.get("source_sha256"), "collection_controls":plan.get("collection_controls"),
             "rows":rows, "training_eligible":False, "training_authorized":False, "sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return payload


def write_e318_replenishment_manifest(*, summary: dict[str, Any], next_round: str, output: Path) -> dict[str, Any]:
    prior={"R1":"R0","R2":"R1"}
    if next_round not in prior or summary.get("round") != prior[next_round]:
        raise ValueError("E3.18 recovery must proceed R0 -> R1 -> R2")
    if output.exists():
        raise ValueError("E3.18 replenishment destination is immutable")
    controls=summary.get("collection_controls")
    if not isinstance(controls,dict) or controls.get("uncached_input_token_cap") != E318_TOKEN_CAP or controls.get("transport_image_max_side") != 512:
        raise ValueError("E3.18 recovery has no frozen 512px controls")
    work=[]
    for row in summary.get("rows") or []:
        if row.get("delivery_status") not in DELIVERY_FAILURES: continue
        rec=recovery_attempt(prior_attempt_ordinal=int(row.get("attempt_ordinal",-1)),status=str(row["delivery_status"]),predecessor_request_id=str(row.get("request_id") or ""))
        if rec.get("new_attempt"):
            work.append({**{key:row[key] for key in ("sample_id","image_group_id","route_progression")},
                         "work_id":f"{next_round}:{row['sample_id']}:e318-all-unknown-512-rag","round":next_round,**rec})
    payload={"schema_version":"agrinet.e318-all-unknown-512-rag-audit-manifest/v1","protocol":E318_PROTOCOL,
             "campaign_id":summary.get("campaign_id"),"round":next_round,"audit_only":True,"all_simulated_unknown":True,
             "work_items":work,"workers":4,"micu_intent_limit":8000,"automatic_replay_allowed":False,
             "collection_controls":controls,"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return payload


def e318_final_report(*, source_rows: list[dict[str, Any]], summaries: list[dict[str, Any],],
                      token_budget: dict[str, Any], provider_usage: dict[str, int]) -> dict[str, Any]:
    """Close the exact R0/R1/R2 lineage without treating delivery as quality."""
    if len(source_rows) != 32 or any(row.get("arm") != "simulated_unknown" for row in source_rows):
        raise ValueError("E3.18 final report requires its 32-row all-Unknown source")
    if [summary.get("round") for summary in summaries] != ["R0", "R1", "R2"]:
        raise ValueError("E3.18 final report requires frozen R0/R1/R2 summaries")
    by_id={str(row.get("sample_id")):row for row in source_rows}
    latest={str(row.get("sample_id")):row for row in summaries[0].get("rows") or []}
    if set(latest) != set(by_id):
        raise ValueError("E3.18 R0 scope does not equal source")
    recovery_scopes=[]
    for summary in summaries[1:]:
        expected={sid for sid,row in latest.items() if row.get("delivery_status") in DELIVERY_FAILURES}
        observed={str(row.get("sample_id")) for row in summary.get("rows") or []}
        if observed != expected:
            raise ValueError("E3.18 recovery summary scope is not exact")
        recovery_scopes.append({"round":summary["round"], "rows":len(observed)})
        latest.update({str(row["sample_id"]):row for row in summary.get("rows") or []})
    folds=Counter(); cells=Counter(); routes=Counter(); delivery=Counter(); terminal=Counter(); contracts=Counter()
    for sid, result in latest.items():
        source=by_id[sid]; folds[str(_fold(source))]+=1
        cells[f"{source['question_type']}/{source['task_domain']}"]+=1
        routes[str(result.get("final_route") or "none")]+=1
        status=str(result.get("delivery_status") or "missing"); delivery[status]+=1
        if status in {*DELIVERY_FAILURES, "budget_shortfall"}: category="delivery_shortfall"
        elif result.get("winner"): category="accepted_rag_refusal" if result.get("rag_result") == "compliant_refusal" else "accepted"
        else: category=str(result.get("quality_status") or "quality_rejected")
        terminal[category]+=1
        if result.get("contract_error"): contracts[str(result["contract_error"])]+=1
    cap=int(token_budget.get("uncached_input_token_cap", 0))
    committed=int(token_budget.get("committed_uncached_input_tokens", 0))
    if cap != E318_TOKEN_CAP or committed > cap:
        raise ValueError("E3.18 fixed token budget is invalid or exceeded")
    prompt=int(provider_usage.get("prompt_tokens", 0)); cached=int(provider_usage.get("cached_tokens", 0))
    return {
        "schema_version": "agrinet.e318-all-unknown-512-rag-audit-final-report/v1",
        "protocol": E318_PROTOCOL, "rows_target":32, "rows_closed":len(latest),
        "all_simulated_unknown":True, "classifier_fold_counts":dict(folds),
        "question_domain_counts":dict(cells), "final_attempt_routes":dict(routes),
        "latest_delivery_statuses":dict(delivery), "terminal_counts":dict(terminal),
        "contract_errors":dict(contracts), "rounds":["R0", "R1", "R2"],
        "recovery_scopes":[{"round":"R0", "rows":32}, *recovery_scopes], "no_r3":True,
        "transport_image_max_side":512, "token_budget":token_budget,
        "provider_usage":{**provider_usage, "cache_rate_of_prompt_tokens": (cached / prompt if prompt else 0.0)},
        "budget_comparison_note":"E3.17 and E3.18 use different identity-disjoint sources; this is an operational budget observation, not a paired causal estimate.",
        "training_eligible":False, "training_authorized":False, "sft_may_start":False,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--prior", type=Path, action="append", required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args=parser.parse_args(argv)
    if args.source_output.exists() or args.manifest_output.exists():
        raise ValueError("E3.18 preparation destinations are immutable")
    candidates=[json.loads(line) for line in args.candidate_source.read_text(encoding="utf-8").splitlines() if line.strip()]
    prior=[json.loads(line) for path in args.prior for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source=materialize_e318_source(select_e318_all_unknown(candidates, prior_rows=prior))
    args.source_output.parent.mkdir(parents=True, exist_ok=True)
    args.source_output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in source), encoding="utf-8")
    manifest=write_e318_manifest(source=args.source_output, campaign_id=args.campaign_id, output=args.manifest_output)
    print(json.dumps({"rows": len(source), "prior_rows": len(prior),
                      "transport_image_max_side": manifest["collection_controls"]["transport_image_max_side"],
                      "uncached_input_token_cap": manifest["collection_controls"]["uncached_input_token_cap"],
                      "training_authorized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
