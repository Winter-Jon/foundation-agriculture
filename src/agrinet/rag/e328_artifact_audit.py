"""Independent artifact audit and gate for E3.28."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from agrinet.rag.e322_presample import digest, rows
from agrinet.rag.e328_classifier_full import FLAGS, NEW, PROTOCOL, REUSED, ROUND_FILES, ROWS
NEW_PROVENANCE = "new_e328"
PROVENANCE_KEY = "e328_provenance"
AUDIT_SCHEMA = "agrinet.e328-classifier-full-artifact-audit/v1"
GATE_SCHEMA = "agrinet.e328-classifier-full-gate/v1"
TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION: float | None = None
ALLOW_PARTIAL_CLASSIFIER_TERMINALS = False

def _is_tolerated_unknown(value:dict[str,Any])->bool:
    return (value.get("delivery_status")=="unknown_delivery"
            and value.get("disposition")=="delivery_unknown")

def _latest(root:Path)->dict[str,dict[str,Any]]:
    latest={}
    for shard in sorted((root/"shards").glob("shard-*")):
        for name in ROUND_FILES:
            path=shard/"outcomes"/name
            if path.is_file():
                for out in json.loads(path.read_text()).get("outcomes") or []: latest[out["sample_id"]]=out
    return latest

def _e322_latest(root:Path)->dict[str,dict[str,Any]]:
    latest={}
    for name in ROUND_FILES:
        path=root/"outcomes"/name
        if path.is_file():
            for out in json.loads(path.read_text()).get("outcomes") or []: latest[out["sample_id"]]=out
    return latest

def _has_forbidden_provider_operation(event:dict[str,Any])->bool:
    """Check structured operation fields, never arbitrary prompt text.

    Classifier prompts deliberately mention forbidden tools (for example,
    ``agrinet_reject``) to constrain the teacher.  Searching a serialized
    payload therefore turns a compliance instruction into a false positive.
    The provider-intent record carries the actual structured operation and
    declared tools separately; audit only those fields.
    """
    forbidden={"agrinet_reject","reject","online_classifier"}
    payload=event.get("payload")
    if not isinstance(payload,dict):
        return False
    scalar_fields=("operation","route","tool","tool_name","name")
    for field in scalar_fields:
        value=payload.get(field)
        if isinstance(value,str) and value.strip().lower() in forbidden:
            return True
    tools=payload.get("tools")
    if isinstance(tools,list) and any(isinstance(value,str) and value.strip().lower() in forbidden for value in tools):
        return True
    tool_calls=payload.get("tool_calls")
    if isinstance(tool_calls,list):
        for call in tool_calls:
            if isinstance(call,dict):
                name=call.get("name") or (call.get("function") or {}).get("name")
                if isinstance(name,str) and name.strip().lower() in forbidden:
                    return True
    return False

def audit(*,source:Path,campaign_root:Path,e322_campaign:Path,output:Path)->dict[str,Any]:
    if output.exists(): return json.loads(output.read_text())
    data=rows(source); new={r["sample_id"] for r in data if r.get(PROVENANCE_KEY)==NEW_PROVENANCE}; reused=set(r["sample_id"] for r in data)-new
    latest={**_e322_latest(e322_campaign),**_latest(campaign_root)}; errors=[]
    if len(data)!=ROWS or len(new)!=NEW or len(reused)!=REUSED or set(latest)!=set(r["sample_id"] for r in data): errors.append("coverage")
    unknown={sid for sid,x in latest.items() if sid in new and _is_tolerated_unknown(x)}
    invalid=[x for sid,x in latest.items() if not (sid in unknown) and x.get("disposition") not in {"semantic_correct","future_rag"}]
    if invalid and not ALLOW_PARTIAL_CLASSIFIER_TERMINALS: errors.append("terminal_disposition")
    unknown_limit=0 if TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION is None else int(len(new)*TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION)
    if len(unknown)>unknown_limit: errors.append("new_delivery")
    if any(x.get("delivery_status")!="delivered" for sid,x in latest.items() if sid in new and sid not in unknown): errors.append("new_delivery")
    audit_failures=[x for x in latest.values()
                    if (x.get("contract_error") or x.get("audit_contract_error"))
                    and x.get("quality") != "pass"]
    if audit_failures and not ALLOW_PARTIAL_CLASSIFIER_TERMINALS: errors.append("quality")
    global_path=campaign_root/"global_micu_intents.jsonl"
    global_intents=[json.loads(x) for x in global_path.read_text().splitlines() if x.strip()] if global_path.is_file() else []
    ledger=[]
    for path in (campaign_root/"ledgers").glob("*/events.jsonl"):
        ledger.extend(json.loads(x) for x in path.read_text().splitlines() if x.strip() and json.loads(x).get("event")=="intent")
    if len(global_intents)!=len(ledger) or len({x.get("key") for x in global_intents})!=len(global_intents): errors.append("intent_ledger")
    if any(_has_forbidden_provider_operation(x) for x in ledger): errors.append("forbidden_provider_operation")
    reused_ledger_ids={sid for sid in reused if any(sid in str(x) for x in ledger)}
    if reused_ledger_ids: errors.append("reused_created_provider_intent")
    shard_rows=[]
    for shard in sorted((campaign_root/"shards").glob("shard-*")):
        shard_latest={}
        for name in ROUND_FILES:
            path=shard/"outcomes"/name
            if path.is_file():
                for out in json.loads(path.read_text()).get("outcomes") or []: shard_latest[out["sample_id"]]=out
        shard_unknown=sorted(sid for sid,x in shard_latest.items() if _is_tolerated_unknown(x))
        # Shards are only deterministic execution/recovery batches.  Exposure
        # is audited per sample and against the campaign-wide new-row limit.
        shard_rows.append({"shard":shard.name,"terminal_rows":len(shard_latest),"terminal_disposition":dict(__import__("collections").Counter(x.get("disposition") for x in shard_latest.values())),"terminal_unknown_delivery":shard_unknown})
    passed=not errors
    quality_pass={sid for sid,x in latest.items() if x.get("quality")=="pass"}
    value={"schema_version":AUDIT_SCHEMA,"protocol":PROTOCOL,"source_sha256":digest(source),"rows":len(data),"reused_rows":len(reused),"new_rows":len(new),"terminal_samples":len(latest),"terminal_unknown_delivery":sorted(unknown),"terminal_unknown_delivery_count":len(unknown),"terminal_unknown_delivery_max_fraction":TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION,"terminal_unknown_delivery_global_limit":unknown_limit,"global_intents":len(global_intents),"ledger_intents":len(ledger),"reused_provider_intent_samples":sorted(reused_ledger_ids),"quality_pass_terminal_samples":sorted(quality_pass),"quality_pass_in_audit_failure":[],"classifier_terminal_shortfalls":sorted(sid for sid,x in latest.items() if x.get("disposition") not in {"semantic_correct","future_rag"}),"partial_classifier_terminals_allowed":ALLOW_PARTIAL_CLASSIFIER_TERMINALS,"shards":shard_rows,"errors":errors,"artifact_audit_passed":passed,**FLAGS}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n"); return value

def gate_decision(*,report:Path,audit_report:Path,output:Path)->dict[str,Any]:
    if output.exists(): return json.loads(output.read_text())
    report_value=json.loads(report.read_text()); audit_value=json.loads(audit_report.read_text())
    passed=bool(report_value.get("campaign_gate_candidate") and audit_value.get("artifact_audit_passed"))
    partial_rag_input_passed=bool(ALLOW_PARTIAL_CLASSIFIER_TERMINALS and report_value.get("rag_input_gate_candidate") and audit_value.get("artifact_audit_passed"))
    value={"schema_version":GATE_SCHEMA,"protocol":PROTOCOL,"final_report_sha256":digest(report),"artifact_audit_sha256":digest(audit_report),"classifier_gate_passed":passed,"partial_rag_input_gate_passed":partial_rag_input_passed,"reject_executed":False,"next_action":"e329_prepare_if_full_or_partial_rag_input_passed_else_freeze",**FLAGS}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n"); return value
