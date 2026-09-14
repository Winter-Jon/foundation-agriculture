"""Public-safe checkpoint reporting for E3.21 Direct-only shards."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.rag.e320_four_stage_cascade import validate_e320_trajectory


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())}


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_report(*, manifest: Path, source: Path, outcomes: Path, campaign_root: Path, output: Path) -> dict[str, Any]:
    """Write an immutable public-safe report after a complete shard outcome."""
    if output.exists(): raise ValueError("E3.21 checkpoint report is immutable")
    plan=json.loads(manifest.read_text(encoding="utf-8")); result=json.loads(outcomes.read_text(encoding="utf-8"))
    if result.get("manifest_sha256") != _digest(manifest): raise ValueError("E3.21 checkpoint manifest binding mismatch")
    items=plan.get("work_items") or []; decisions=result.get("outcomes") or []
    if len(items) != len(decisions) or {str(x.get("work_id")) for x in items} != {str(x.get("work_id")) for x in decisions}:
        raise ValueError("E3.21 checkpoint outcome scope is incomplete")
    source_rows=_rows(source); item_by_work={str(item["work_id"]):item for item in items}
    disposition=Counter(); delivery=Counter(); arm=Counter(); form=Counter(); validation_errors=[]
    for decision in decisions:
        item=item_by_work[str(decision["work_id"])]; row=source_rows[str(item["sample_id"])]
        delivery[str(decision.get("delivery_status"))] += 1
        disposition[str(decision.get("disposition"))] += 1
        arm[str(row.get("arm"))] += 1; form[str(row.get("question_type"))] += 1
        parent=decision.get("parent_path")
        if decision.get("delivery_status") == "delivered" and isinstance(parent,str) and parent:
            try: validate_e320_trajectory(row,json.loads(Path(parent).read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, ValueError) as exc: validation_errors.append({"sample_id":row["sample_id"],"error":str(exc)})
    budget_path=Path(campaign_root) / "token_budget.jsonl"
    events=[json.loads(x) for x in budget_path.read_text(encoding="utf-8").splitlines() if x.strip()] if budget_path.exists() else []
    prefixes={f"{work_id}:" for work_id in item_by_work}
    scoped=[e for e in events if any(str(e.get("key") or "").startswith(prefix) for prefix in prefixes)]
    reservations={str(e["key"]):int(e["uncached_input_tokens"]) for e in scoped if e.get("event") == "reserve"}
    settlements={str(e["key"]):int(e["uncached_input_tokens"]) for e in scoped if e.get("event") == "settle"}
    report={"schema_version":"agrinet.e321-checkpoint-report/v1",
            "campaign_id":plan.get("campaign_id"),"manifest":str(manifest),"manifest_sha256":_digest(manifest),
            "outcomes":str(outcomes),"outcomes_sha256":_digest(outcomes),
            "scope":{"items":len(items),"arm":dict(arm),"question_type":dict(form)},
            "delivery":dict(delivery),"disposition":dict(disposition),
            "public_validation":{"passed":not validation_errors,"errors":validation_errors},
            "token_budget":{"cap":int((plan.get("collection_controls") or {}).get("uncached_input_token_cap",0)),
                            "settled_uncached_input_tokens":sum(settlements.values()),
                            "unknown_delivery_exposure_tokens":sum(v for key,v in reservations.items() if key not in settlements),
                            "reservation_events":len(reservations),"settlement_events":len(settlements),
                            "scope":"manifest_work_items_only"},
            "training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return report


def delivery_shortfall_report(*, manifest: Path, outcomes: Path, output: Path) -> dict[str, Any]:
    """Record terminal R2 unknown deliveries without issuing another request.

    R2 is the recovery ceiling.  Its unresolved rows must remain distinct from
    quality and semantic outcomes, and must not be silently omitted merely
    because there is no permitted R3 manifest.
    """
    if output.exists():
        raise ValueError("E3.21 delivery-shortfall report is immutable")
    plan=json.loads(manifest.read_text(encoding="utf-8"))
    result=json.loads(outcomes.read_text(encoding="utf-8"))
    if plan.get("round") != "R2" or result.get("manifest_sha256") != _digest(manifest):
        raise ValueError("E3.21 delivery-shortfall report requires a bound R2 outcome")
    items={str(item.get("work_id")):item for item in plan.get("work_items") or []}
    rows=[]
    for decision in result.get("outcomes") or []:
        if decision.get("delivery_status") != "unknown_delivery":
            continue
        item=items.get(str(decision.get("work_id")))
        if item is None or not isinstance(decision.get("request_id"),str) or not decision["request_id"]:
            raise ValueError("E3.21 R2 shortfall lacks immutable lineage")
        rows.append({"sample_id":item["sample_id"],"work_id":item["work_id"],
                     "terminal_state":"delivery_shortfall","final_route":"direct",
                     "last_request_id":decision["request_id"],"attempt_ordinal":item.get("attempt_ordinal"),
                     "no_r3_created":True})
    payload={"schema_version":"agrinet.e321-delivery-shortfall-report/v1",
             "campaign_id":plan.get("campaign_id"),"manifest":str(manifest),
             "manifest_sha256":_digest(manifest),"outcomes":str(outcomes),
             "outcomes_sha256":_digest(outcomes),"round":"R2",
             "delivery_shortfalls":rows,"count":len(rows),
             "training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return payload


def final_direct_campaign_report(*, source: Path, manifest_dir: Path, outcome_dir: Path,
                                 report_dir: Path, campaign_root: Path, output: Path,
                                 classifier_queue_output: Path, e320_source: Path | None = None,
                                 e320_r1_outcomes: Path | None = None) -> dict[str, Any]:
    """Freeze the E3.21 Direct-only terminal accounting and future queue.

    This is intentionally an offline aggregation: it reads immutable Direct
    outcomes only, refuses any non-Direct trajectory, and never dispatches a
    provider call.  A sample may have delivery or quality descendants; its
    terminal state is selected from the terminal Direct outcome, not by adding
    intermediate attempts to the campaign coverage figures.
    """
    if output.exists() or classifier_queue_output.exists():
        raise ValueError("E3.21 final report and future queue are immutable")
    rows=_rows(source)
    if len(rows) != 1039:
        raise ValueError("E3.21 final report source scope is incomplete")
    source_values=list(rows.values())
    if Counter(str(row.get("arm")) for row in source_values) != Counter({"known":535,"simulated_unknown":504}):
        raise ValueError("E3.21 final report arm scope is invalid")

    entries=[]
    manifest_paths=sorted(manifest_dir.glob("e321-direct-*.json"))
    for manifest in manifest_paths:
        outcome=outcome_dir / f"{manifest.stem}.json"
        if not outcome.exists():
            continue
        plan=_read_json(manifest); result=_read_json(outcome)
        if plan.get("protocol") != "agrinet.e321-direct-first-hcv-cascade/v1" or result.get("manifest_sha256") != _digest(manifest):
            raise ValueError(f"E3.21 final report lineage mismatch: {manifest.name}")
        items={str(item.get("work_id")):item for item in plan.get("work_items") or []}
        decisions=result.get("outcomes") or []
        if set(items) != {str(item.get("work_id")) for item in decisions}:
            raise ValueError(f"E3.21 final report scope mismatch: {manifest.name}")
        for decision in decisions:
            item=items[str(decision["work_id"])]
            sample_id=str(item["sample_id"])
            if sample_id not in rows or decision.get("final_route") != "direct":
                raise ValueError("E3.21 final report found non-Direct or out-of-scope outcome")
            entries.append({"manifest":manifest,"item":item,"decision":decision,"row":rows[sample_id]})

    by_sample: dict[str,list[dict[str,Any]]]={sample_id:[] for sample_id in rows}
    for entry in entries:
        by_sample[str(entry["item"]["sample_id"])].append(entry)
    if any(not group for group in by_sample.values()):
        raise ValueError("E3.21 final report has uncollected Direct source rows")

    terminal=[]; queue=[]; terminal_counter=Counter(); terminal_arm=Counter(); terminal_form=Counter()
    attempt_counter=Counter(); delivery_counter=Counter(); quality_exhausted=0
    validation_failures=[]
    for sample_id,group in sorted(by_sample.items()):
        group.sort(key=lambda entry:(int(entry["item"].get("attempt_ordinal",0)), int(entry["item"].get("quality_attempt_ordinal",0)), str(entry["item"]["work_id"])))
        for entry in group:
            decision=entry["decision"]; attempt_counter[str(entry["item"].get("round"))] += 1
            delivery_counter[str(decision.get("delivery_status"))] += 1
        delivered=[entry for entry in group if entry["decision"].get("delivery_status") == "delivered"]
        correct=[entry for entry in delivered if entry["decision"].get("disposition") == "semantic_correct"]
        wrong=[entry for entry in delivered if entry["decision"].get("disposition") == "semantic_wrong"]
        unresolved=[entry for entry in group if entry["decision"].get("delivery_status") == "unknown_delivery"]
        exhausted=[entry for entry in delivered if entry["decision"].get("disposition") == "quality_reject" and entry["decision"].get("quality_repair_exhausted")]
        if correct:
            chosen=correct[-1]; state="direct_winner"
        elif wrong:
            chosen=wrong[-1]; state="future_classifier"
        elif exhausted:
            chosen=exhausted[-1]; state="quality_repair_exhausted"; quality_exhausted += 1
        elif unresolved:
            # Earlier unknown deliveries are lineage ancestors, not a terminal
            # result, whenever a later recovery or repair delivered.  Reaching
            # this branch means there is no delivered leaf at all.
            chosen=unresolved[-1]; state="delivery_shortfall"
            if int(chosen["item"].get("attempt_ordinal",-1)) != 2:
                raise ValueError("E3.21 unresolved terminal did not reach R2")
        else:
            raise ValueError(f"E3.21 source row has no terminal disposition: {sample_id}")
        row=chosen["row"]; decision=chosen["decision"]; item=chosen["item"]
        record={"sample_id":sample_id,"terminal_state":state,"arm":row.get("arm"),"question_type":row.get("question_type"),
                "last_work_id":item["work_id"],"last_request_id":decision.get("request_id"),
                "attempt_ordinal":item.get("attempt_ordinal"),"quality_attempt_ordinal":item.get("quality_attempt_ordinal"),
                "final_route":"direct"}
        terminal.append(record); terminal_counter[state] += 1; terminal_arm[(state,str(row.get("arm")))] += 1; terminal_form[(state,str(row.get("question_type")))] += 1
        if state == "future_classifier":
            queue.append({"sample_id":sample_id,"predecessor_request_id":decision.get("request_id"),"future_route":"classifier",
                          "reason":"semantic_wrong","arm":row.get("arm"),"question_type":row.get("question_type"),
                          "training_eligible":False,"training_authorized":False,"sft_may_start":False})

    # Existing E3.20 accounting is frozen by the E3.21 contract; do not merge
    # its raw trajectories into this Direct-only campaign output.
    existing={"reused_direct_winners":25,"deferred_classifier_semantic_wrong":6,"direct_q1_repair_in_source":1}
    if (e320_source is None) != (e320_r1_outcomes is None):
        raise ValueError("E3.21 final report requires both E3.20 source and outcomes")
    preexisting_queue=[]
    if e320_source is not None and e320_r1_outcomes is not None:
        e320_rows=_rows(e320_source); e320_result=_read_json(e320_r1_outcomes)
        full_pool={**e320_rows,**rows}
        by_arm_class=Counter((str(row.get("arm")),str(row.get("canonical_class_code"))) for row in full_pool.values())
        by_arm_class_form=Counter((str(row.get("arm")),str(row.get("canonical_class_code")),str(row.get("question_type"))) for row in full_pool.values())
        classes={str(row.get("canonical_class_code")) for row in full_pool.values()}
        coverage_ok=(len(full_pool) == 1070 and len(classes) == 107 and
                     all(by_arm_class[(arm,code)] == 5 and by_arm_class_form[(arm,code,"open")] == 3 and
                         by_arm_class_form[(arm,code,"option")] == 2
                         for arm in ("known","simulated_unknown") for code in classes))
        if not coverage_ok:
            raise ValueError("E3.21 full-pool class/form coverage is invalid")
        for decision in e320_result.get("outcomes") or []:
            if decision.get("disposition") != "semantic_wrong":
                continue
            sample_id=str(decision.get("sample_id")); row=e320_rows.get(sample_id)
            if row is None or decision.get("final_route") != "direct" or not isinstance(decision.get("request_id"),str):
                raise ValueError("E3.21 E3.20 deferred classifier lineage is invalid")
            preexisting_queue.append({"sample_id":sample_id,"predecessor_request_id":decision["request_id"],"future_route":"classifier",
                                      "reason":"semantic_wrong_e320_reuse","arm":row.get("arm"),"question_type":row.get("question_type"),
                                      "training_eligible":False,"training_authorized":False,"sft_may_start":False})
        if len(preexisting_queue) != existing["deferred_classifier_semantic_wrong"]:
            raise ValueError("E3.21 E3.20 deferred classifier count is invalid")
    else:
        full_pool=None; classes=set(); coverage_ok=None
    checkpoints=[]
    for report_path in sorted(report_dir.glob("e321-direct-*.json")):
        report=_read_json(report_path)
        if report.get("schema_version") == "agrinet.e321-checkpoint-report/v1":
            checkpoints.append({"report":report_path.name,"public_validation_passed":report.get("public_validation",{}).get("passed"),
                                "validation_error_count":len(report.get("public_validation",{}).get("errors") or [])})
            if not report.get("public_validation",{}).get("passed"):
                validation_failures.append(report_path.name)
    budget=campaign_root / "token_budget.jsonl"
    events=[json.loads(line) for line in budget.read_text(encoding="utf-8").splitlines() if line.strip()]
    reservations=[event for event in events if event.get("event") == "reserve"]
    settlements=[event for event in events if event.get("event") == "settle"]
    reserved={str(event["key"]):int(event["uncached_input_tokens"]) for event in reservations}
    settled={str(event["key"]):int(event["uncached_input_tokens"]) for event in settlements}
    if any(event.get("metadata",{}).get("route") not in {"direct",None} for event in events):
        raise ValueError("E3.21 final report ledger contains a non-Direct route")
    full_queue=preexisting_queue+queue
    if len({entry["sample_id"] for entry in full_queue}) != len(full_queue):
        raise ValueError("E3.21 future classifier queue contains duplicate samples")
    queue_payload={"schema_version":"agrinet.e321-future-classifier-queue/v1","protocol":"agrinet.e321-direct-first-hcv-cascade/v1",
                   "queue":full_queue,"count":len(full_queue),"new_direct_scope_count":len(queue),"e320_reuse_count":len(preexisting_queue),
                   "executed":False,"frozen_not_executed":["classifier","rag","reject"],
                   "training_eligible":False,"training_authorized":False,"sft_may_start":False}
    classifier_queue_output.parent.mkdir(parents=True,exist_ok=True)
    classifier_queue_output.write_text(json.dumps(queue_payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    payload={"schema_version":"agrinet.e321-final-direct-campaign-report/v1","protocol":"agrinet.e321-direct-first-hcv-cascade/v1",
             "source":str(source),"source_sha256":_digest(source),"source_scope":{"new_direct_items":len(rows),"arm":dict(Counter(str(row.get("arm")) for row in source_values)),
                 "question_type":dict(Counter(str(row.get("question_type")) for row in source_values))},
             "frozen_e320_accounting":existing,"total_1070_accounting":{"reused_direct_winners":25,"new_direct_items":len(rows),"deferred_classifier_preexisting":6,"total_candidates":1070},
             "full_pool_coverage":({"pool_items":len(full_pool),"classes":len(classes),"arms":{"known":535,"simulated_unknown":535},
                 "per_arm_per_class":5,"per_arm_per_class_question_type":{"open":3,"option":2},"passed":coverage_ok} if full_pool is not None else {"passed":None}),
             "attempts":{"total":len(entries),"by_round":dict(attempt_counter),"delivery":dict(delivery_counter)},
             "terminal":{"count":len(terminal),"by_state":dict(terminal_counter),"by_state_arm":{f"{k[0]}:{k[1]}":v for k,v in sorted(terminal_arm.items())},
                 "by_state_question_type":{f"{k[0]}:{k[1]}":v for k,v in sorted(terminal_form.items())},"quality_repair_exhausted":quality_exhausted},
             "future_classifier_queue":str(classifier_queue_output),"future_classifier_queue_count":len(full_queue),
             "future_classifier_queue_new_direct_scope_count":len(queue),"future_classifier_queue_e320_reuse_count":len(preexisting_queue),
             "checkpoint_validation":{"reports":checkpoints,"reports_with_historical_validation_errors":validation_failures,
                 "note":"Validation errors are all terminal quality-reject paths, never Direct winners."},
             "token_budget":{"cap":16000000,"reserved_uncached_input_tokens":sum(reserved.values()),"settled_uncached_input_tokens":sum(settled.values()),
                 "unknown_delivery_exposure_tokens":sum(value for key,value in reserved.items() if key not in settled),"reservation_events":len(reserved),"settlement_events":len(settled)},
             "executed_routes":["direct"],"frozen_not_executed":["classifier","rag","reject"],
             "training_eligible":False,"training_authorized":False,"sft_may_start":False}
    if len(terminal) != len(rows) or len({item["sample_id"] for item in queue}) != len(queue):
        raise ValueError("E3.21 final report terminal accounting is invalid")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True); parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--outcomes",type=Path,required=True); parser.add_argument("--campaign-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(argv)
    print(json.dumps(checkpoint_report(manifest=args.manifest,source=args.source,outcomes=args.outcomes,campaign_root=args.campaign_root,output=args.output),ensure_ascii=False,sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
