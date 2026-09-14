"""Dry-run gate for the prospective E3.20 four-stage collector.

No provider execution path exists in this adapter.  It deliberately proves the
frozen inputs and lineage before a separately authorized live binding is added.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from argparse import Namespace
from typing import Any

from agrinet.common.credentials import yunwu_environment
from agrinet.rag.e320_collect import collect_manifest
from agrinet.rag.e320_collect import _stage
from agrinet.rag.e320_four_stage_cascade import E320_PROTOCOL, STAGES
from agrinet.rag.e321_direct_first import CAP as E321_CAP, E321_PROTOCOL, RESERVATION as E321_RESERVATION, SCHEMA as E321_SCHEMA
from agrinet.rag.e320_private import e320_private_audit_request, parse_e320_private_audit
from agrinet.rag.e35_cascade_collect import _TOOLS, _append_native_tool_exchange, _initial_messages, _ledger_payload, _request, _validate_tool_arguments
from agrinet.rag.e35_classifier_cascade import public_classifier_card
from agrinet.rag.e35_ledger import E35Ledger
from agrinet.rag.e35_budget import E35TokenBudget, uncached_input_tokens
from agrinet.rag.e35_private import private_trajectory_projection
from agrinet.rag.e35_transport import transport_image
from agrinet.rag.micu_classifier_hcv_v2 import local_rag_health
from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget, execute_rag, parse_teacher_action
from agrinet.research.hcv.collector import post_teacher_json
from agrinet.research.hcv.v13_collector import _isolated_micu_request


def _private_names(path: Path) -> dict[str, str]:
    """Read a frozen private canonical registry or a full frozen label map."""
    raw=json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not isinstance(raw,list):
        raise ValueError("E3.20 private registry is malformed")
    names={str(item.get("canonical_code") or item.get("canonical_class_code")):str(item.get("canonical_english_name") or item.get("english_name") or "") for item in raw if isinstance(item,dict)}
    if not names or any(code == "None" or not name for code,name in names.items()):
        raise ValueError("E3.20 private registry is malformed")
    return names


def validate_e320_live_inputs(*, manifest: Path, source: Path) -> dict[str, object]:
    plan=json.loads(manifest.read_text(encoding="utf-8"))
    is_e321 = plan.get("schema_version") == E321_SCHEMA and plan.get("protocol") == E321_PROTOCOL
    if not is_e321 and (plan.get("schema_version") != "agrinet.e320-four-stage-cascade-manifest/v1" or plan.get("protocol") != E320_PROTOCOL):
        raise ValueError("E3.20 live manifest schema is invalid")
    if any(plan.get(flag) is not False for flag in ("training_eligible","training_authorized","sft_may_start")):
        raise ValueError("E3.20 live manifest improperly authorizes training")
    actual=hashlib.file_digest(source.open("rb"),"sha256").hexdigest()
    if plan.get("source_sha256") != actual:
        raise ValueError("E3.20 live source binding mismatch")
    controls=plan.get("collection_controls") or {}
    reserve=controls.get("reservation_uncached_tokens") or {}
    expected_cap = E321_CAP if is_e321 else 8_000_000
    expected_reserve = E321_RESERVATION if is_e321 else {"generation":{"direct":5000,"classifier":8000,"rag":12000,"reject":5000},"private_audit":18000}
    if (controls.get("transport_image_max_side") != 512 or controls.get("uncached_input_token_cap") != expected_cap
            or controls.get("max_public_turns_per_route") != 12 or controls.get("max_rag_searches") != 3
            or reserve != expected_reserve):
        raise ValueError("E3.20 live collection controls are not frozen")
    rows={str(json.loads(line).get("sample_id")):json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()}
    items=plan.get("work_items") or []
    if not items or len({str(item.get("work_id")) for item in items}) != len(items):
        raise ValueError("E3.20 live work scope is invalid")
    for item in items:
        if str(item.get("sample_id")) not in rows: raise ValueError("E3.20 live work item is absent from source")
        _stage(item)
        if is_e321 and item.get("resume_route") != "direct":
            raise ValueError("E3.21 only authorizes Direct collection")
    return {"ready":True,"rows":len(items),"routes":list(STAGES),"source_sha256":actual,"provider_requests":0,"sft_may_start":False}


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--rag-endpoint")
    parser.add_argument("--private-registry",type=Path)
    parser.add_argument("--teacher-model",default="gpt-5.6-terra")
    parser.add_argument("--timeout",type=int,default=180)
    parser.add_argument("--budget-path",type=Path)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv)
    result=validate_e320_live_inputs(manifest=args.manifest,source=args.source)
    if not args.dry_run:
        is_e321 = result.get("source_sha256") and json.loads(args.manifest.read_text(encoding="utf-8")).get("protocol") == E321_PROTOCOL
        needed=(args.output,args.output_root,args.private_registry,args.budget_path)
        if not all(needed) or (not is_e321 and not args.rag_endpoint):
            raise ValueError("collection requires output, output-root, private registry, budget path, and RAG endpoint unless E3.21 Direct-only")
        if args.output.exists():
            raise ValueError("E3.20 outcome destination is immutable")
        if not is_e321 and local_rag_health(args.rag_endpoint).get("status") != "ok":
            raise ValueError("E3.20 local RAG health preflight failed")
        registry=_private_names(args.private_registry)
        source_rows=[json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(str((row.get("private") or {}).get("truth_code") or "") not in registry for row in source_rows):
            raise ValueError("E3.20 private registry does not cover frozen source truth")
        environment=yunwu_environment(profile="micu_slb")
        base_url=environment.get("YUNWU_API_BASE_URL")
        if not isinstance(base_url,str) or not base_url.startswith("https://"):
            raise ValueError("E3.20 teacher endpoint is missing or not HTTPS")
        headers={"Authorization":f"Bearer {environment['YUNWU_API_KEY']}","Content-Type":"application/json"}
        post_args=Namespace(teacher_retries=0,teacher_timeout=args.timeout,teacher_retry_sleep=0.0)
        controls=json.loads(args.manifest.read_text(encoding="utf-8"))["collection_controls"]
        token_budget=E35TokenBudget(args.budget_path,uncached_input_token_cap=int(controls["uncached_input_token_cap"]))
        global_budget=GlobalMicuBudget(args.output_root / "global_micu_intents.jsonl",limit=int(json.loads(args.manifest.read_text(encoding="utf-8"))["micu_intent_limit"]))

        def budgeted_call(*, ledger: E35Ledger, kind: str, key: str, payload: dict[str, Any], invoke, route: str, turn: int = 1):
            reserve_kind="private_audit" if kind == "private_audit" else "generation"
            allowance=int(controls["reservation_uncached_tokens"][reserve_kind] if reserve_kind == "private_audit" else controls["reservation_uncached_tokens"]["generation"][route])
            budget_key=f"{ledger.binding['work_id']}:{kind}:{turn}"
            token_budget.reserve(budget_key,uncached_input_tokens=allowance,metadata={"route":route,"kind":kind,"turn":turn})
            global_budget.reserve(budget_key)
            request_id,response=ledger.call(kind=kind,key=key,payload=payload,invoke=invoke)
            observed=uncached_input_tokens(response)
            if observed is not None: token_budget.settle(budget_key,uncached_input_tokens=observed)
            return request_id,response

        def teacher(payload: dict[str, Any]) -> dict[str, Any]:
            # The ordinary HTTP helper converts network failures to
            # UnknownTeacherDelivery, but a blocked TLS/socket poll can evade
            # that helper's timeout in a worker thread.  Keep the provider
            # call in a killable child so a lost delivery reaches the existing
            # E35 ledger as unresolved rather than stalling an entire immutable
            # Direct shard indefinitely.  No retry occurs here: the E3.21
            # controller alone may create predecessor-bound R1/R2 attempts.
            return _isolated_micu_request(
                base_url.rstrip("/") + "/chat/completions", payload, headers, int(args.timeout),
            )

        def stage_runner(row: dict[str, Any], stage: str, context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            ledger=E35Ledger(args.output_root / "ledgers" / str(context["work_id"]).replace(":","_"),work_id=str(context["work_id"]),attempt_ordinal=int(context["attempt_ordinal"]),intent_limit=8000)
            if stage == "reject":
                prior=context.get("prior_rag_trajectory")
                request={"model":args.teacher_model,"temperature":0.0,"top_p":1.0,"max_tokens":4096,"messages":[{"role":"system","content":row["teacher_system_prompts"]["reject"]},{"role":"user","content":json.dumps({"prior_rag_trajectory":private_trajectory_projection(prior)},ensure_ascii=False)}]}
                request_id,raw=budgeted_call(ledger=ledger,kind="generation",key="reject:generation:1",payload={"sample_id":row["sample_id"],"stage":stage},invoke=lambda:teacher(request),route=stage)
                action=parse_teacher_action(raw)
                if action.get("type") != "final": raise ValueError("E3.20 Reject must end without tools")
                return request_id,{"answer":action["content"],"tool_trace":[],"messages":request["messages"]+[{"role":"assistant","content":action["content"]}]}
            messages=_initial_messages(row,stage,transport_max_side=int(controls["transport_image_max_side"]))
            trace=[]; predicted=False; rag_called=False
            for turn in range(1,int(controls["max_public_turns_per_route"])+1):
                request=_request(row,stage,messages,args.teacher_model,4096,rag_called=rag_called)
                request_id,raw=budgeted_call(ledger=ledger,kind="generation",key=f"{stage}:generation:{turn}",payload=_ledger_payload(request,sample_id=row["sample_id"],route=stage,turn=turn),invoke=lambda:teacher(request),route=stage,turn=turn)
                action=parse_teacher_action(raw)
                if action.get("type") == "final":
                    return request_id,{"answer":action["content"],"tool_calls":[event["call"] for event in trace],"tool_trace":trace,"messages":messages+[{"role":"assistant","content":action["content"]}]}
                name,args_tool=action.get("name"),action.get("arguments")
                _validate_tool_arguments(str(name),args_tool,predicted=predicted)
                if stage == "direct" or (stage == "classifier" and name == "agrinet_rag_search"):
                    raise ValueError("E3.20 stage tool contract violation")
                if name == "agrinet_classifier_predict":
                    result_tool=public_classifier_card(row); predicted=True
                elif name == "agrinet_classifier_expand":
                    result_tool=public_classifier_card(row,expanded=True)
                elif name == "agrinet_rag_search":
                    if stage != "rag" or not predicted or rag_called: raise ValueError("E3.20 RAG tool order is invalid")
                    result_tool=execute_rag(args.rag_endpoint,{"image_path":row["image_path"]},args_tool); rag_called=True
                else: raise ValueError("E3.20 unknown tool")
                trace.append({"call":{"name":name,"arguments":args_tool},"response":result_tool})
                _append_native_tool_exchange(messages,raw,action,result_tool)
            raise ValueError("E3.20 stage exhausted turn limit")

        def audit(row: dict[str, Any], trajectory: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
            private=row.get("private") or {}
            code=str(private.get("truth_code") or "")
            if code not in registry: raise ValueError("E3.20 source truth is absent from registry")
            image,_=transport_image(Path(row["image_path"]),max_side=int(controls["transport_image_max_side"]))
            request=e320_private_audit_request(truth_name=registry[code],correct_option=private.get("correct_option"),public_options=row.get("public_options") or [],trajectory=private_trajectory_projection(trajectory),model=args.teacher_model,image=image)
            ledger=E35Ledger(args.output_root / "ledgers" / str(context["work_id"]).replace(":","_"),work_id=str(context["work_id"]),attempt_ordinal=int(context["attempt_ordinal"]),intent_limit=8000)
            request_id,raw=budgeted_call(ledger=ledger,kind="private_audit",key=f"{context['stage']}:private-audit",payload={"sample_id":row["sample_id"],"stage":context["stage"]},invoke=lambda:teacher(request),route=str(context["stage"]))
            try:
                return parse_e320_private_audit(raw)
            except ValueError as exc:
                # A malformed private verdict cannot safely become a quality
                # rejection.  Preserve the delivered ledger event and let the
                # controller create only a predecessor-bound R1/R2 attempt.
                raise DeliveryUnresolved("E3.20 private audit verdict is unresolved", request_id=request_id) from exc

        result=collect_manifest(manifest=args.manifest,source=args.source,output=args.output,output_root=args.output_root,stage_runner=stage_runner,private_audit=audit)
        print(json.dumps({"round":result["round"],"outcomes":len(result["outcomes"]),"sft_may_start":False},ensure_ascii=False))
        return 0
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
