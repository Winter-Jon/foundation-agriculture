"""Independent artifact audit and final gate for E3.29."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from agrinet.rag.e322_presample import digest,rows
from agrinet.rag.e328_classifier_full import FLAGS
from agrinet.rag.e329_visual_top3_full import PROTOCOL

ROUND_FILES=("r0.json","q1.json","r1.json","q1-r1.json","r2.json","q1-r2.json")

def _latest(root:Path)->dict[str,dict[str,Any]]:
    latest={}
    for shard in sorted((root/"shards").glob("shard-*")):
        for name in ROUND_FILES:
            path=shard/"outcomes"/name
            if path.is_file():
                for out in json.loads(path.read_text()).get("outcomes") or []: latest[out["sample_id"]]=out
    return latest

def audit(*,source:Path,campaign_root:Path,output:Path)->dict[str,Any]:
    if output.exists(): return json.loads(output.read_text())
    data=rows(source); latest=_latest(campaign_root); errors=[]; rag_evidence=0; slot_loss=[]; reject_calls=0
    if set(latest)!={r["sample_id"] for r in data}: errors.append("coverage")
    safe_dispositions={"semantic_correct","future_reject"}
    residuals={sid:out for sid,out in latest.items() if out.get("disposition") not in safe_dispositions}
    for sid,out in latest.items():
        if out.get("disposition") not in safe_dispositions:
            # A delivered post-Q1 quality failure is a sample-level residual.
            # It cannot provide RAG evidence, but cannot freeze unrelated rows.
            if not (out.get("disposition")=="quality_reject" and out.get("delivery_status")=="delivered"):
                errors.append(f"terminal:{sid}")
            continue
        evidence=Path(str(out.get("evidence_path") or "")); trajectory=Path(str(out.get("trajectory_path") or ""))
        try:
            raw=json.loads(evidence.read_text()); names=raw.get("returned_standard_class_names") or []
            if raw.get("tool")!="agrinet_rag_search" or raw.get("arguments",{}).get("query")!="visual morphology" or raw.get("arguments",{}).get("retrieval_type")!="visual" or raw.get("arguments",{}).get("top_k")!=3 or "ranker" in raw.get("arguments",{}): errors.append(f"retrieval:{sid}")
            if len(names)!=3 or any(not isinstance(x,str) or not x.strip() for x in names): slot_loss.append(sid)
            trace=json.loads(trajectory.read_text()).get("tool_trace") or []; calls=[x.get("call",{}).get("name") for x in trace]
            if calls!=["agrinet_classifier_predict","agrinet_rag_search"]: errors.append(f"tools:{sid}")
            reject_calls+=calls.count("agrinet_reject"); rag_evidence+=1
        except Exception as exc: errors.append(f"state:{sid}:{type(exc).__name__}")
    if slot_loss: errors.append("candidate_slot_loss")
    global_path=campaign_root/"global_micu_intents.jsonl"
    global_intents=[json.loads(x) for x in global_path.read_text().splitlines() if x.strip()] if global_path.is_file() else []
    ledger=[]
    for path in (campaign_root/"ledgers").glob("*/events.jsonl"):
        ledger.extend(json.loads(x) for x in path.read_text().splitlines() if x.strip() and json.loads(x).get("event")=="intent")
    if len(global_intents)!=len(ledger) or len({x.get("key") for x in global_intents})!=len(global_intents): errors.append("intent_ledger")
    value={"schema_version":"agrinet.e329-visual-top3-rag-full-artifact-audit/v1","protocol":PROTOCOL,"source_sha256":digest(source),"rows":len(data),"terminal_samples":len(latest),"safe_terminal_samples":sorted(set(latest)-set(residuals)),"safe_terminal_count":len(latest)-len(residuals),"rag_terminal_residuals":[{"sample_id":sid,"disposition":out.get("disposition"),"delivery_status":out.get("delivery_status"),"contract_error":out.get("contract_error"),"audit_contract_error":out.get("audit_contract_error"),**FLAGS} for sid,out in sorted(residuals.items())],"rag_evidence_samples":rag_evidence,"candidate_slot_loss_samples":sorted(slot_loss),"global_intents":len(global_intents),"ledger_intents":len(ledger),"reject_calls":reject_calls,"errors":errors,"artifact_audit_passed":not errors and reject_calls==0,**FLAGS}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n"); return value

def gate_decision(*,report:Path,audit_report:Path,output:Path)->dict[str,Any]:
    if output.exists(): return json.loads(output.read_text())
    report_value=json.loads(report.read_text()); audit_value=json.loads(audit_report.read_text()); passed=bool(report_value.get("campaign_gate_candidate") and audit_value.get("artifact_audit_passed"))
    partial_passed=bool(report_value.get("rag_input_gate_candidate") and audit_value.get("artifact_audit_passed"))
    value={"schema_version":"agrinet.e329-visual-top3-rag-full-gate/v1","protocol":PROTOCOL,"final_report_sha256":digest(report),"artifact_audit_sha256":digest(audit_report),"rag_gate_passed":passed,"partial_rag_input_gate_passed":partial_passed,"reject_executed":False,"next_action":"interview_user",**FLAGS}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n"); return value
