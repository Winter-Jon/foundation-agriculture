"""Independent artifact audit and immutable gate for E3.23."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
from agrinet.rag.e323_rag_presample import PROTOCOL, digest, read_jsonl

def _json_sha(value:Any)->str:
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def audit(*,source:Path,campaign_root:Path,output:Path)->dict[str,Any]:
    if output.exists(): return json.loads(output.read_text())
    rows=read_jsonl(source); errors=[]; trajectories=0; rag_receipts=0; predict_trajectories=0; leaked=[]
    latest={}
    for path in sorted((campaign_root/"outcomes").glob("*.json")):
        for outcome in json.loads(path.read_text()).get("outcomes") or []: latest[outcome["sample_id"]]=outcome
    terminal_paths={str(value.get("parent_path")) for value in latest.values() if value.get("parent_path")}
    for path in sorted((campaign_root/"public").glob("*/trajectory.json")):
        raw=path.read_text(); trajectories+=1
        if "data:image" in raw or "e323_prior_group" in raw or "truth_code" in raw: leaked.append(str(path))
        tr=json.loads(raw); calls=[x.get("call",{}).get("name") for x in tr.get("tool_trace") or []]
        valid=calls.count("agrinet_classifier_predict")==1 and 1<=calls.count("agrinet_rag_search")<=3 and calls[0]=="agrinet_classifier_predict"
        if valid: predict_trajectories+=1
        elif str(path) in terminal_paths: errors.append(f"terminal_tool_order:{path}")
        if "agrinet_reject" in calls: errors.append(f"reject_called:{path}")
    for path in sorted((campaign_root/"resolver-receipts").glob("*/*.json")):
        item=json.loads(path.read_text())
        if item.get("tool")=="agrinet_rag_search":
            rag_receipts+=1
            if _json_sha(item.get("public_result"))!=item.get("public_result_sha256"): errors.append(f"receipt_hash:{path}")
    terminal_valid=sum(1 for value in latest.values() if value.get("parent_path") and str(value.get("parent_path")) not in {e.split(":",1)[1] for e in errors if e.startswith("terminal_tool_order:")})
    payload={"schema_version":"agrinet.e323-rag-preflight-artifact-audit/v1","protocol":PROTOCOL,"source_sha256":digest(source),"source_rows":len(rows),"public_trajectories":trajectories,"valid_predict_rag_trajectories":predict_trajectories,"terminal_samples":len(latest),"terminal_valid_tool_order":terminal_valid,"rag_receipts":rag_receipts,"private_leakage_paths":leaked,"errors":errors,"artifact_audit_passed":bool(len(rows)==16 and len(latest)==16 and terminal_valid==16 and rag_receipts>=16 and not leaked and not errors),"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n"); return payload

def gate_decision(*,report:Path,audit_report:Path,output:Path)->dict[str,Any]:
    if output.exists(): return json.loads(output.read_text())
    rep=json.loads(report.read_text()); aud=json.loads(audit_report.read_text())
    passed=bool(rep.get("campaign_gate_candidate") and aud.get("artifact_audit_passed"))
    payload={"schema_version":"agrinet.e323-rag-preflight-gate-decision/v1","protocol":PROTOCOL,"final_report_sha256":digest(report),"artifact_audit_sha256":digest(audit_report),"presample_gate_passed":passed,"next_action":"interview_user_before_reject_or_expansion","reject_executed":False,"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n"); return payload
