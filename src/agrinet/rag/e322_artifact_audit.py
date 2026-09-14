"""Independent post-run audit for frozen E3.22 campaign artifacts."""
from __future__ import annotations

import argparse, json
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.rag.e322_campaign import validate_trajectory
from agrinet.rag.e322_presample import IDENTITIES, LEGACY_PROTOCOL, PREVIOUS_PROTOCOL, PROTOCOL, digest, rows

ROUND_FILES=("r0.json","q1.json","r1.json","q1-r1.json","r2.json","q1-r2.json")
PRIVATE_MARKERS=("checkpoint_sha256","held_out_fold","label_map_codes","truth_code","training_manifest","/data/")

def audit(*,source:Path,campaign_root:Path,output:Path)->dict[str,Any]:
    if output.exists(): raise ValueError("E3.22 artifact audit is immutable")
    source_rows=rows(source); by={x["sample_id"]:x for x in source_rows}
    docs=[json.loads((campaign_root/"outcomes"/name).read_text()) for name in ROUND_FILES]
    protocols={x.get("protocol") for x in docs}
    if len(protocols)!=1 or not protocols<={PROTOCOL,PREVIOUS_PROTOCOL,LEGACY_PROTOCOL}: raise ValueError("E3.22 outcome protocol mismatch")
    protocol=next(iter(protocols)); strict=protocol==PROTOCOL
    latest={}; attempts=[]
    for doc in docs:
        for item in doc.get("outcomes") or []: latest[item["sample_id"]]=item; attempts.append(item)
    if set(latest)!=set(by): raise ValueError("E3.22 audit lineage does not close 32 rows")
    leaks=[]; image_payloads=0; tool_errors=0; trajectory_errors=[]; terminal_trajectory_errors=[]; delivered_trajectories=0; resolver_receipts=0; resolver_receipt_errors=0
    seen_trajectory_sha=set()
    for item in attempts:
        parent=item.get("parent_path")
        if not parent: continue
        trajectory_path=Path(parent); trajectory_sha=digest(trajectory_path)
        trajectory=json.loads(trajectory_path.read_text()); delivered_trajectories+=1
        rendered=json.dumps(trajectory,ensure_ascii=False)
        leaks.extend({"work_id":item["work_id"],"marker":marker} for marker in PRIVATE_MARKERS if marker in rendered)
        image_payloads+=int("data:image" in rendered)
        calls=[x.get("call",{}).get("name") for x in trajectory.get("tool_trace") or []]
        if calls not in (["agrinet_classifier_predict"],["agrinet_classifier_predict","agrinet_classifier_expand"]): tool_errors+=1
        for event in trajectory.get("tool_trace") or []:
            receipt=event.get("resolver") if isinstance(event,dict) else None
            if isinstance(receipt,dict) and trajectory_sha not in seen_trajectory_sha:
                resolver_receipts+=1
                expected=__import__("hashlib").sha256(json.dumps(event.get("response"),ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
                resolver_receipt_errors+=int(receipt.get("kind")!="frozen_card" or receipt.get("public_card_sha256")!=expected)
            elif strict and trajectory_sha not in seen_trajectory_sha: resolver_receipt_errors+=1
        try: validate_trajectory(by[item["sample_id"]],trajectory)
        except ValueError as exc:
            failure={"work_id":item["work_id"],"error":str(exc)}; trajectory_errors.append(failure)
            if latest.get(item["sample_id"],{}).get("work_id")==item["work_id"]: terminal_trajectory_errors.append(failure)
        seen_trajectory_sha.add(trajectory_sha)
    durable_receipts=list((campaign_root/"resolver-receipts").glob("*/*.json"))
    if strict:
        for path in durable_receipts:
            try:
                value=json.loads(path.read_text())
                if value.get("kind")!="frozen_card" or value.get("tool") not in {"agrinet_classifier_predict","agrinet_classifier_expand"}: resolver_receipt_errors+=1
            except (OSError,json.JSONDecodeError): resolver_receipt_errors+=1
    lineage=[]; lineage_binding_failures=0
    for name,doc in zip(ROUND_FILES,docs):
        if name=="r0.json": continue
        manifest_path=campaign_root/"manifests"/name
        if not manifest_path.is_file():
            if strict: lineage_binding_failures+=len(doc.get("outcomes") or [])
            continue
        manifest=json.loads(manifest_path.read_text()); work={x["work_id"]:x for x in manifest.get("work_items") or []}
        if doc.get("manifest_sha256")!=digest(manifest_path): lineage_binding_failures+=1
        for item in doc.get("outcomes") or []:
            child=work.get(item["work_id"]); ok=bool(child and item.get("work_item_sha256")==__import__("hashlib").sha256(json.dumps(child,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest() and item.get("predecessor_request_id")==child.get("predecessor_request_id") and item.get("predecessor_outcome_sha256")==child.get("predecessor_outcome_sha256") and item.get("predecessor_manifest_sha256")==child.get("predecessor_manifest_sha256"))
            lineage_binding_failures+=int(not ok)
            lineage.append({"work_id":item["work_id"],"predecessor_request_id":child.get("predecessor_request_id") if child else None,"request_id":item.get("request_id"),"new_request_id":bool(child and item.get("request_id") and item["request_id"]!=child.get("predecessor_request_id")),"binding_valid":ok,"delivery_status":item["delivery_status"]})
    ledger_events=[]
    for path in (campaign_root/"ledgers").glob("*/events.jsonl"):
        ledger_events.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    classifier_network_intents=sum(1 for x in ledger_events if x.get("event")=="intent" and "classifier" in str(x.get("kind","")) and x.get("kind")!="generation")
    final=Counter(x["disposition"] for x in latest.values()); arm_correct=Counter(by[sid]["arm"] for sid,x in latest.items() if x.get("winner"))
    new_id_failures=sum(1 for x in lineage if x["request_id"] and not x["new_request_id"])
    artifact_pass=bool(strict and not leaks and not image_payloads and not tool_errors and not terminal_trajectory_errors and not resolver_receipt_errors and resolver_receipts>0 and not classifier_network_intents and not lineage_binding_failures and not new_id_failures)
    result={"schema_version":"agrinet.e322-classifier-presample-artifact-audit/v4","protocol":protocol,"strict_v3_audit":strict,"rounds":[x["round"] for x in docs],"rows":32,"identity_unique":{key:len({x[key] for x in source_rows})==32 for key in IDENTITIES},"truth_classes":len({x["canonical_class_code"] for x in source_rows}),"delivered_trajectories":delivered_trajectories,"private_binding_leaks":leaks,"private_binding_leak_count":len(leaks),"persisted_input_image_trajectory_count":image_payloads,"tool_order_errors":tool_errors,"trajectory_contract_errors":trajectory_errors,"trajectory_contract_error_count":len(trajectory_errors),"terminal_trajectory_contract_errors":terminal_trajectory_errors,"terminal_trajectory_contract_error_count":len(terminal_trajectory_errors),"resolver_receipt_count":resolver_receipts,"durable_resolver_receipt_count":len(durable_receipts),"resolver_receipt_errors":resolver_receipt_errors,"online_classifier_requests":classifier_network_intents,"lineage":lineage,"lineage_binding_failures":lineage_binding_failures,"lineage_new_request_id_failures":new_id_failures,"r0_contract_errors":sum(1 for x in docs[0]["outcomes"] if x.get("contract_error")),"r0_private_quality_failures":sum(1 for x in docs[0]["outcomes"] if x.get("quality")=="fail"),"final_disposition":dict(final),"semantic_correct_by_arm":dict(arm_correct),"quality_exhausted":sum(1 for x in latest.values() if x.get("disposition")=="quality_reject"),"budget_shortfall":sum(1 for x in latest.values() if x.get("delivery_status")=="budget_shortfall"),"artifact_gate_passed":artifact_pass,"presample_gate_passed":False,"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n"); return result

def gate_decision(*,report:Path,audit_report:Path,output:Path)->dict[str,Any]:
    if output.exists(): return json.loads(output.read_text())
    final=json.loads(report.read_text()); artifact=json.loads(audit_report.read_text())
    if final.get("protocol")!=PROTOCOL or artifact.get("protocol")!=PROTOCOL: raise ValueError("E3.22 final gate requires v3 inputs")
    passed=bool(final.get("campaign_gate_candidate") and artifact.get("artifact_gate_passed"))
    result={"schema_version":"agrinet.e322-classifier-presample-gate-decision/v1","protocol":PROTOCOL,"final_report_sha256":digest(report),"artifact_audit_sha256":digest(audit_report),"campaign_gate_candidate":bool(final.get("campaign_gate_candidate")),"artifact_gate_passed":bool(artifact.get("artifact_gate_passed")),"presample_gate_passed":passed,"next_action":"interview_user","training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n"); return result

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--campaign-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(argv); result=audit(source=a.source,campaign_root=a.campaign_root,output=a.output); print(json.dumps({k:result[k] for k in ("private_binding_leak_count","tool_order_errors","trajectory_contract_error_count","presample_gate_passed")},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
