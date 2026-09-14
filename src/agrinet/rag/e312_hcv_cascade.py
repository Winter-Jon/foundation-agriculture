"""E3.12: normalize a provider's pure-Hermes-plan/native-call wire combination."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
from agrinet.rag.e311_hcv_cascade import (PROMPTS, select_e311_audit, select_e310_coverage,
    validate_e311_trajectory)
from agrinet.rag.e35_classifier_cascade import (ARMS, DELIVERY_FAILURES, ROUTES,
    recovery_attempt)

E312_PROTOCOL = "agrinet.e312-hcv-cascade/v1"

def validate_e312_trajectory(row:dict[str,Any], trajectory:dict[str,Any])->None:
    try: validate_e311_trajectory(row,trajectory)
    except ValueError as exc: raise ValueError(str(exc).replace("E3.11","E3.12")) from exc

def materialize_e312_source(rows:list[dict[str,Any]], coverage:dict[str,str])->list[dict[str,Any]]:
    result=[]
    for row in rows:
        private=dict(row.get("private") or {})
        if row["sample_id"] in coverage: private["e312_route_coverage"]=coverage[row["sample_id"]].rsplit(":",1)[1]
        result.append({**row,"e39_protocol":E312_PROTOCOL,"teacher_system_prompts":dict(PROMPTS),"private":private})
    return result

def write_manifest(source:Path,campaign_id:str,output:Path)->dict[str,Any]:
    if output.exists(): raise ValueError("E3.12 manifest is immutable")
    rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    if len(rows)!=32 or any(r.get("e39_protocol")!=E312_PROTOCOL for r in rows): raise ValueError("E3.12 source binding invalid")
    p={"schema_version":"agrinet.e312-hcv-cascade-manifest/v1","protocol":E312_PROTOCOL,"campaign_id":campaign_id,"round":"R0","source":str(source),"source_sha256":hashlib.file_digest(source.open("rb"),"sha256").hexdigest(),"source_rows_expected":32,"audit_only":True,"work_items":[{"work_id":f"R0:{r['sample_id']}:e312-hcv","round":"R0","sample_id":r["sample_id"],"image_group_id":r.get("image_group_id",r["image_sha256"]),"attempt_ordinal":0,"predecessor_request_id":None,"route_progression":["direct","classifier","rag"]} for r in rows],"workers":4,"micu_intent_limit":8000,"automatic_replay_allowed":False,"collection_controls":{"uncached_input_token_cap":300000,"transport_image_max_side":1024,"max_public_turns_per_route":12,"max_rag_searches":3,"reservation_uncached_tokens":{"generation":{"direct":5000,"classifier":8000,"rag":12000},"private_audit":18000}},"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(p,ensure_ascii=False,indent=2,sort_keys=True)+"\n"); return p

def write_e312_replenishment_manifest(*, summary:dict[str,Any], next_round:str, output:Path)->dict[str,Any]:
    """Create the exact, delivery-only E3.12 R1/R2 recovery scope."""
    previous = {"R1": "R0", "R2": "R1"}
    if next_round not in previous or summary.get("round") != previous[next_round]:
        raise ValueError("E3.12 recovery must proceed R0 -> R1 -> R2")
    if output.exists():
        raise ValueError("E3.12 replenishment destination is immutable")
    controls = summary.get("collection_controls")
    if not isinstance(controls,dict) or controls.get("max_rag_searches") != 3 or controls.get("max_public_turns_per_route") != 12:
        raise ValueError("E3.12 recovery has no frozen collection controls")
    work=[]
    for row in summary.get("rows",[]):
        if row.get("delivery_status") not in DELIVERY_FAILURES:
            continue
        rec=recovery_attempt(prior_attempt_ordinal=int(row.get("attempt_ordinal",-1)), status=str(row["delivery_status"]), predecessor_request_id=str(row.get("request_id") or ""))
        if rec.get("new_attempt"):
            work.append({**{key:row[key] for key in ("sample_id","image_group_id","route_progression")},
                         "work_id":f"{next_round}:{row['sample_id']}:e312-hcv","round":next_round,**rec})
    payload={"schema_version":"agrinet.e312-hcv-cascade-manifest/v1","protocol":E312_PROTOCOL,
             "campaign_id":summary.get("campaign_id"),"round":next_round,"audit_only":True,
             "work_items":work,"workers":4,"micu_intent_limit":8000,
             "automatic_replay_allowed":False,"collection_controls":controls,
             "training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    return payload

def e312_layered_audit_report(*, source_rows:list[dict[str,Any]], summaries:list[dict[str,Any]])->dict[str,Any]:
    """Validate complete R0/R1/R2 lineage and report the E3.12 protocol gate."""
    if len(source_rows)!=32 or [s.get("round") for s in summaries] != ["R0","R1","R2"]:
        raise ValueError("E3.12 final audit needs its 32-row R0/R1/R2 lineage")
    by_id={str(r.get("sample_id")):r for r in source_rows}
    if len(by_id)!=32:
        raise ValueError("E3.12 audit source has duplicate IDs")
    latest={str(r.get("sample_id")):r for r in summaries[0].get("rows",[])}
    if set(latest)!=set(by_id):
        raise ValueError("E3.12 R0 summary does not close source scope")
    for summary in summaries[1:]:
        expected={sid for sid,row in latest.items() if row.get("delivery_status") in DELIVERY_FAILURES}
        observed={str(row.get("sample_id")) for row in summary.get("rows",[])}
        if expected != observed:
            raise ValueError("E3.12 recovery summary scope is not exact")
        latest.update({str(row["sample_id"]):row for row in summary.get("rows",[])})
    from collections import Counter
    terminal,routes=Counter(),Counter(); coverage={arm:set() for arm in ARMS}; rag_evidence_closed=False
    for sample_id,row in latest.items():
        routes[str(row.get("final_route") or "none")]+=1
        if row.get("delivery_status") in DELIVERY_FAILURES:
            terminal["delivery_shortfall"]+=1
        elif row.get("winner"):
            terminal["accepted"]+=1
            target=(by_id[sample_id].get("private") or {}).get("e312_route_coverage")
            if target==row.get("final_route"):
                coverage[by_id[sample_id]["arm"]].add(target)
            if row.get("final_route")=="rag":
                try:
                    trace=json.loads(Path(str(row.get("parent_path") or "")).read_text()).get("tool_trace",[])
                    rag_evidence_closed |= any(event.get("call",{}).get("name")=="agrinet_rag_search" and isinstance(event.get("response"),dict) for event in trace if isinstance(event,dict))
                except (OSError,json.JSONDecodeError):
                    pass
        else:
            terminal[str(row.get("quality_status") or "quality_rejected")]+=1
    coverage_winners={arm:sorted(routes_) for arm,routes_ in coverage.items()}
    observed=any(row.get("delivery_status")=="delivered" for row in latest.values())
    gate=bool(observed and rag_evidence_closed and all(set(coverage_winners[arm])==set(ROUTES) for arm in ARMS))
    return {"schema_version":"agrinet.e312-audit-final-report/v1","protocol":E312_PROTOCOL,
            "rows":32,"rounds":["R0","R1","R2"],"terminal_counts":dict(terminal),
            "final_routes":dict(routes),"coverage_winners":coverage_winners,
            "protocol_observed":observed,"rag_evidence_closed":rag_evidence_closed,
            "protocol_gate_passed":gate,"full_campaign_remaining_candidates":942,
            "full_campaign_expansion":"requires_separate_manifest" if gate else "not_authorized",
            "training_eligible":False,"training_authorized":False,"sft_may_start":False}

def main(argv:list[str]|None=None)->int:
 import argparse
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="operation")
 prep=sub.add_parser("prepare-audit"); prep.add_argument("--candidate-source",type=Path,required=True); prep.add_argument("--prior",action="append",type=Path,required=True); prep.add_argument("--coverage-output",type=Path,required=True); prep.add_argument("--source-output",type=Path,required=True); prep.add_argument("--campaign-id",required=True); prep.add_argument("--manifest-output",type=Path,required=True)
 replenish=sub.add_parser("plan-replenishment"); replenish.add_argument("--summary",type=Path,required=True); replenish.add_argument("--next-round",choices=("R1","R2"),required=True); replenish.add_argument("--output",type=Path,required=True)
 report=sub.add_parser("audit-final-report"); report.add_argument("--source",type=Path,required=True); report.add_argument("--r0-summary",type=Path,required=True); report.add_argument("--r1-summary",type=Path,required=True); report.add_argument("--r2-summary",type=Path,required=True); report.add_argument("--output",type=Path,required=True)
 if argv and argv[0].startswith("--"): argv=["prepare-audit",*argv]
 a=p.parse_args(argv)
 if a.operation=="plan-replenishment":
  m=write_e312_replenishment_manifest(summary=json.loads(a.summary.read_text()),next_round=a.next_round,output=a.output); print(json.dumps({"round":m["round"],"rows":len(m["work_items"])})); return 0
 if a.operation=="audit-final-report":
  if a.output.exists(): raise ValueError("E3.12 audit report destination is immutable")
  result=e312_layered_audit_report(source_rows=[json.loads(line) for line in a.source.read_text().splitlines() if line.strip()],summaries=[json.loads(path.read_text()) for path in (a.r0_summary,a.r1_summary,a.r2_summary)])
  a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n"); print(json.dumps({"protocol_gate_passed":result["protocol_gate_passed"]})); return 0 if result["protocol_gate_passed"] else 2
 if any(x.exists() for x in (a.coverage_output,a.source_output,a.manifest_output)): raise ValueError("E3.12 preparation destinations are immutable")
 candidates=[json.loads(x) for x in a.candidate_source.read_text().splitlines() if x.strip()]; prior=[json.loads(x) for path in a.prior for x in path.read_text().splitlines() if x.strip()]
 rows=select_e311_audit(candidates,prior_rows=prior,seed="e312-audit-v1"); cov=select_e310_coverage(rows,seed="e312-coverage-v1"); src=materialize_e312_source(rows,cov)
 a.coverage_output.parent.mkdir(parents=True,exist_ok=True); a.coverage_output.write_text(json.dumps(cov,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
 a.source_output.parent.mkdir(parents=True,exist_ok=True); a.source_output.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in src))
 m=write_manifest(a.source_output,a.campaign_id,a.manifest_output); print(json.dumps({"rows":32,"prior_rows":len(prior),"coverage_rows":len(cov),"manifest_rows":len(m["work_items"])}))
 return 0
if __name__=="__main__": raise SystemExit(main())
