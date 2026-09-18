"""Independent artifact gate for E3.24 structured RAG."""
from __future__ import annotations
import json
from pathlib import Path
from agrinet.rag.e324_contract import parse_json_response,render,validate_closure,validate_plan
from agrinet.rag.e324_structured_rag import PROTOCOL,digest,rows
from agrinet.rag.e35_classifier_cascade import public_classifier_card
from agrinet.rag.e324_campaign import _catalog,_extend_e327_rag_catalog,_extend_rag_catalog

def audit(*,source:Path,campaign_root:Path,output:Path):
    if output.exists(): return json.loads(output.read_text())
    data=rows(source); protocol=data[0].get("e39_protocol") if data else None; expected=28 if protocol in {"agrinet.e325-structured-rag/v1","agrinet.e326-structured-rag/v1","agrinet.e327-visual-top3-rag/v1"} else 8
    by={r["sample_id"]:r for r in data}; latest={}; errors=[]; leaks=[]; valid=0; rag_evidence_samples=0; reject_calls=0
    for name in ("r0.json","q1.json","r1.json","q1-r1.json","r2.json","q1-r2.json"):
        p=campaign_root/"outcomes"/name
        if p.exists():
            for x in json.loads(p.read_text()).get("outcomes") or []: latest[x["sample_id"]]=x
    closure_responses={}
    for ledger_path in (campaign_root/"ledgers").glob("*/events.jsonl"):
        for line in ledger_path.read_text().splitlines():
            event=json.loads(line)
            if event.get("event")=="result" and event.get("status")=="delivered":
                closure_responses[event.get("request_id")]=event.get("response")
    for sid,x in latest.items():
        evidence_candidate=Path(x["evidence_path"]) if x.get("evidence_path") else Path(str(x.get("state_dir") or ""))/"rag-evidence.json"
        if evidence_candidate.is_file():
            frozen_evidence=json.loads(evidence_candidate.read_text())
            if frozen_evidence.get("tool")=="agrinet_rag_search": rag_evidence_samples+=1
        if x.get("disposition") not in {"semantic_correct","future_reject"}: continue
        try:
            plan_path=Path(x["plan_path"]); evidence_path=Path(x["evidence_path"]); trajectory_path=Path(x["trajectory_path"])
            for path,key in ((plan_path,"plan_sha256"),(evidence_path,"evidence_sha256"),(trajectory_path,"trajectory_sha256")):
                if digest(path)!=x[key]: errors.append(f"hash:{sid}:{key}")
            raw="\n".join(path.read_text() for path in (plan_path,evidence_path,trajectory_path))
            if any(k in raw for k in ("data:image","truth_code","e324_stratum","e325_stratum","e326_stratum","e327_stratum","checkpoint_sha256","/data/","image_path")): leaks.append(str(trajectory_path))
            catalog=_catalog(by[sid],public_classifier_card(by[sid])); planner_ids=set(catalog); evidence=json.loads(evidence_path.read_text())
            _extend_e327_rag_catalog(catalog,evidence) if protocol=="agrinet.e327-visual-top3-rag/v1" else _extend_rag_catalog(catalog,evidence)
            raw_plan=json.loads(plan_path.read_text())
            if protocol=="agrinet.e327-visual-top3-rag/v1":
                from agrinet.rag.e327_contract import validate_plan as validate327_plan
                plan=validate327_plan(raw_plan)
            elif protocol=="agrinet.e326-structured-rag/v1":
                from agrinet.rag.e326_contract import validate_plan as validate326_plan
                plan=validate326_plan(raw_plan,by[sid],set(_catalog(by[sid],public_classifier_card(by[sid])).values()))
            else: plan=validate_plan(raw_plan,planner_ids)
            frozen_closure=json.loads((trajectory_path.parent/"closure.json").read_text())
            if protocol=="agrinet.e327-visual-top3-rag/v1":
                from agrinet.rag.e327_contract import parse_json_response as parse327,render as render327,validate_closure as validate327_closure
                raw_closure=parse327(closure_responses[x.get("closure_request_id")])
            elif protocol=="agrinet.e326-structured-rag/v1":
                from agrinet.rag.e326_contract import parse_json_response as parse326,render as render326,validate_closure as validate326_closure
                raw_closure=parse326(closure_responses[x.get("closure_request_id")])
            else: raw_closure=parse_json_response(closure_responses[x.get("closure_request_id")])
            if protocol=="agrinet.e327-visual-top3-rag/v1": closure=validate327_closure(raw_closure,by[sid],catalog); rebuilt=render327(by[sid],closure,catalog)
            elif protocol=="agrinet.e326-structured-rag/v1": closure=validate326_closure(raw_closure,by[sid],catalog); rebuilt=render326(by[sid],plan,closure,catalog)
            else: closure=validate_closure(raw_closure,by[sid],plan,catalog); rebuilt=render(by[sid],plan,closure,catalog)
            if closure!=frozen_closure: raise ValueError("closure provider response differs from frozen state")
            if json.loads(trajectory_path.read_text())["answer"]!=rebuilt: errors.append(f"render:{sid}")
            if evidence.get("tool")!="agrinet_rag_search": errors.append(f"rag:{sid}")
            calls=[e.get("call",{}).get("name") for e in json.loads(trajectory_path.read_text()).get("tool_trace") or []]
            reject_calls+=calls.count("agrinet_reject")
            if calls!=["agrinet_classifier_predict","agrinet_rag_search"]: errors.append(f"tools:{sid}")
            if protocol=="agrinet.e326-structured-rag/v1":
                args=json.loads(trajectory_path.read_text())["tool_trace"][1]["call"]["arguments"]
                if args.get("retrieval_type")!="balanced" or args.get("top_k")!=8 or args.get("ranker")!="rrf": errors.append(f"retrieval_contract:{sid}")
            if protocol=="agrinet.e327-visual-top3-rag/v1":
                args=json.loads(trajectory_path.read_text())["tool_trace"][1]["call"]["arguments"]
                if args.get("retrieval_type")!="visual" or args.get("top_k")!=3 or "ranker" in args: errors.append(f"retrieval_contract:{sid}")
            valid+=1
        except Exception as exc: errors.append(f"state:{sid}:{type(exc).__name__}")
    global_intents=[json.loads(x) for x in (campaign_root/"global_micu_intents.jsonl").read_text().splitlines()] if (campaign_root/"global_micu_intents.jsonl").exists() else []
    ledger_intents=[]
    for path in (campaign_root/"ledgers").glob("*/events.jsonl"):
        ledger_intents += [json.loads(x) for x in path.read_text().splitlines() if json.loads(x).get("event")=="intent"]
    if len(global_intents)!=len(ledger_intents) or len({x["key"] for x in global_intents})!=len(global_intents) or len({x["request_id"] for x in ledger_intents})!=len(ledger_intents): errors.append("provider_intent_accounting")
    coverage={}
    if expected==28:
        if protocol=="agrinet.e327-visual-top3-rag/v1": from agrinet.rag.e327_visual_top3 import CELL_TARGETS,FOLD_TARGETS,IDENTITIES
        elif protocol=="agrinet.e326-structured-rag/v1": from agrinet.rag.e326_structured_rag import CELL_TARGETS,FOLD_TARGETS,IDENTITIES
        else: from agrinet.rag.e325_structured_rag import CELL_TARGETS,FOLD_TARGETS,IDENTITIES
        manifest=json.loads((source.parent/"manifest-r0.json").read_text()); excluded=[rows(Path(v["path"])) for k,v in manifest["immutable_inputs"].items() if k.startswith("excluded_source_")]; excluded=[r for batch in excluded for r in batch]
        if protocol=="agrinet.e327-visual-top3-rag/v1":
            paired=rows(Path(manifest["immutable_inputs"]["e326_source"]["path"])); paired_by={r["sample_id"]:r for r in paired}
            if set(paired_by)!=set(by): errors.append("paired_e326_sample_ids")
            for key in IDENTITIES:
                if any(sid not in paired_by or by[sid][key]!=paired_by[sid][key] for sid in by): errors.append(f"paired_e326_identity:{key}")
        for key in IDENTITIES:
            if len({r[key] for r in data})!=expected or ({r[key] for r in data}&{r[key] for r in excluded}): errors.append(f"identity:{key}")
        if len({r["canonical_class_code"] for r in data})!=expected: errors.append("truth_classes")
        if __import__("collections").Counter(f"{r['question_type']}/{r['task_domain']}" for r in data)!=__import__("collections").Counter(CELL_TARGETS): errors.append("cells")
        if __import__("collections").Counter(r["classifier"]["held_out_fold"] for r in data)!=__import__("collections").Counter(FOLD_TARGETS): errors.append("folds")
        coverage={"identity_unique":{key:len({r[key] for r in data}) for key in IDENTITIES},
                  "identity_overlap_with_excluded":{key:len({r[key] for r in data}&{r[key] for r in excluded}) for key in IDENTITIES},
                  "truth_classes":len({r["canonical_class_code"] for r in data}),
                  "cells":dict(__import__("collections").Counter(f"{r['question_type']}/{r['task_domain']}" for r in data)),
                  "folds":dict(__import__("collections").Counter(str(r["classifier"]["held_out_fold"]) for r in data)),
                  "checkpoint_shas":len({r["classifier"]["checkpoint_sha256"] for r in data})}
    if rag_evidence_samples!=expected: errors.append("rag_evidence_coverage")
    passed=len(data)==expected and len(latest)==expected and not errors and not leaks
    version="v2" if expected==8 else "v1"; experiment=327 if protocol=="agrinet.e327-visual-top3-rag/v1" else 326 if protocol=="agrinet.e326-structured-rag/v1" else 325 if expected==28 else 324
    value={"schema_version":f"agrinet.e{experiment}-structured-rag-artifact-audit/{version}","protocol":protocol,"source_sha256":digest(source),"terminal_samples":len(latest),"valid_structured_trajectories":valid,"rag_evidence_samples":rag_evidence_samples,"source_coverage":coverage,"global_intents":len(global_intents),"ledger_intents":len(ledger_intents),"errors":errors,"private_leakage_paths":leaks,"artifact_audit_passed":passed,"reject_calls":reject_calls,"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n"); return value

def gate_decision(*,report:Path,audit_report:Path,output:Path):
    if output.exists(): return json.loads(output.read_text())
    r=json.loads(report.read_text()); a=json.loads(audit_report.read_text()); passed=bool(r["campaign_gate_candidate"] and a["artifact_audit_passed"]); protocol=r.get("protocol"); experiment=327 if protocol=="agrinet.e327-visual-top3-rag/v1" else 326 if protocol=="agrinet.e326-structured-rag/v1" else 325 if protocol=="agrinet.e325-structured-rag/v1" else 324
    value={"schema_version":f"agrinet.e{experiment}-structured-rag-gate/{'v2' if experiment==324 else 'v1'}","protocol":r["protocol"],"final_report_sha256":digest(report),"artifact_audit_sha256":digest(audit_report),"presample_gate_passed":passed,"reject_executed":False,"next_action":"interview_user","training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); return value
