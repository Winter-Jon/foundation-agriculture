"""E3.13: identity-disjoint HCV successor with explicit content scaffolding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agrinet.rag.e310_hcv_cascade import select_e310_coverage
from agrinet.rag.e311_hcv_cascade import select_e311_audit, validate_e311_trajectory
from agrinet.rag.e35_classifier_cascade import ARMS, DELIVERY_FAILURES, ROUTES, recovery_attempt

E313_PROTOCOL = "agrinet.e313-hcv-cascade/v1"

BASE = (
    "You are an agricultural visual-diagnosis teacher. Use only the image, public question, and actual public tool responses. "
    "Teach Hypothesize–Contrast–Verify (HCV); never reveal or infer private labels, folds, coverage, or hidden metadata. "
    "Final output is exactly <think>...</think><answer>...</answer>. In <think>, use these six headings once and in order: "
    "Visual observations:, Candidate hypotheses:, Candidate comparison:, Evidence:, Rejected alternatives:, Uncertainty:. "
    "Write three image-grounded observations. Under Rejected alternatives:, write at least two separate clauses in exactly this useful form: `candidate name: rejected because visible trait conflicts with ...`. "
    "Each reason must name a concrete visible trait, never score, rank, name semantics, or hidden information. "
    "Under Uncertainty:, give low, medium, or high confidence plus one evidence limitation. "
)
PROMPTS = {
    "direct": BASE + "No tools. For Open, hypothesize two or three concrete classes. For Option, Candidate comparison: must contain exactly four labeled clauses `A. public class name: ...` through `D. public class name: ...`; assess every option before answering. Answer canonical name for Open and `class name — letter` for Option.",
    "classifier": BASE + "First output one standalone pure <think> planning turn and do not answer or call a tool. Next make exactly one native agrinet_classifier_predict call. After the card, answer or make at most one agrinet_classifier_expand call for one stated ambiguity. For Option, still compare A/B/C/D explicitly in the final Candidate comparison:.",
    "rag": BASE + "First output one standalone pure <think> planning turn, then make exactly one native agrinet_classifier_predict call. After its card, make one to three agrinet_rag_search calls; each must be preceded by a standalone plan naming a different unresolved visual discriminator. Use retrieval to verify candidates, never merge classifier and retrieval scores. For Option, still compare A/B/C/D explicitly in the final Candidate comparison:.",
}

def validate_e313_trajectory(row: dict[str, Any], trajectory: dict[str, Any]) -> None:
    try:
        validate_e311_trajectory(row, trajectory)
    except ValueError as exc:
        raise ValueError(str(exc).replace("E3.11", "E3.13")) from exc
    if row.get("question_type") == "option":
        answer=str(trajectory.get("answer") or "")
        thought=answer.split("Candidate comparison:",1)[1].split("Evidence:",1)[0] if "Candidate comparison:" in answer and "Evidence:" in answer else ""
        options=row.get("public_options") or []
        required=[(str(item.get("label")),str(item.get("name"))) for item in options if isinstance(item,dict)]
        if len(required) == 4 and any(label not in thought or name not in thought for label,name in required):
            raise ValueError("E3.13 Option reasoning must compare every public option")

def materialize_e313_source(rows: list[dict[str, Any]], coverage: dict[str, str]) -> list[dict[str, Any]]:
    if len(rows) != 32 or len(coverage) != 6:
        raise ValueError("E3.13 needs a 32-row source and six private route witnesses")
    result=[]
    for row in rows:
        private=dict(row.get("private") or {})
        if row["sample_id"] in coverage:
            private["e313_route_coverage"] = coverage[row["sample_id"]].rsplit(":", 1)[1]
        result.append({**row, "e39_protocol": E313_PROTOCOL, "teacher_system_prompts": dict(PROMPTS), "private": private})
    return result

def write_e313_manifest(*, source: Path, campaign_id: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("E3.13 manifest is immutable")
    rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if len(rows) != 32 or any(row.get("e39_protocol") != E313_PROTOCOL for row in rows):
        raise ValueError("E3.13 source binding invalid")
    payload={"schema_version":"agrinet.e313-hcv-cascade-manifest/v1","protocol":E313_PROTOCOL,"campaign_id":campaign_id,"round":"R0","source":str(source),"source_sha256":hashlib.file_digest(source.open("rb"),"sha256").hexdigest(),"source_rows_expected":32,"audit_only":True,"work_items":[{"work_id":f"R0:{row['sample_id']}:e313-hcv","round":"R0","sample_id":row["sample_id"],"image_group_id":row.get("image_group_id",row["image_sha256"]),"attempt_ordinal":0,"predecessor_request_id":None,"route_progression":["direct","classifier","rag"]} for row in rows],"workers":4,"micu_intent_limit":8000,"automatic_replay_allowed":False,"collection_controls":{"uncached_input_token_cap":300000,"transport_image_max_side":1024,"max_public_turns_per_route":12,"max_rag_searches":3,"reservation_uncached_tokens":{"generation":{"direct":5000,"classifier":8000,"rag":12000},"private_audit":18000}},"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    return payload

def write_e313_replenishment_manifest(*, summary: dict[str, Any], next_round: str, output: Path) -> dict[str, Any]:
    prior={"R1":"R0","R2":"R1"}
    if next_round not in prior or summary.get("round") != prior[next_round]:
        raise ValueError("E3.13 recovery must proceed R0 -> R1 -> R2")
    if output.exists():
        raise ValueError("E3.13 replenishment destination is immutable")
    controls=summary.get("collection_controls")
    if not isinstance(controls,dict) or controls.get("max_rag_searches") != 3 or controls.get("max_public_turns_per_route") != 12:
        raise ValueError("E3.13 recovery has no frozen collection controls")
    work=[]
    for row in summary.get("rows",[]):
        if row.get("delivery_status") not in DELIVERY_FAILURES:
            continue
        rec=recovery_attempt(prior_attempt_ordinal=int(row.get("attempt_ordinal",-1)),status=str(row["delivery_status"]),predecessor_request_id=str(row.get("request_id") or ""))
        if rec.get("new_attempt"):
            work.append({**{key:row[key] for key in ("sample_id","image_group_id","route_progression")},
                         "work_id":f"{next_round}:{row['sample_id']}:e313-hcv","round":next_round,**rec})
    payload={"schema_version":"agrinet.e313-hcv-cascade-manifest/v1","protocol":E313_PROTOCOL,"campaign_id":summary.get("campaign_id"),"round":next_round,"audit_only":True,"work_items":work,"workers":4,"micu_intent_limit":8000,"automatic_replay_allowed":False,"collection_controls":controls,"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    return payload

def e313_layered_audit_report(*, source_rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if len(source_rows) != 32 or [item.get("round") for item in summaries] != ["R0","R1","R2"]:
        raise ValueError("E3.13 final audit needs its 32-row R0/R1/R2 lineage")
    by_id={str(row.get("sample_id")):row for row in source_rows}
    if len(by_id) != 32:
        raise ValueError("E3.13 audit source has duplicate IDs")
    latest={str(row.get("sample_id")):row for row in summaries[0].get("rows",[])}
    if set(latest) != set(by_id):
        raise ValueError("E3.13 R0 summary does not close source scope")
    for summary in summaries[1:]:
        expected={sample_id for sample_id,row in latest.items() if row.get("delivery_status") in DELIVERY_FAILURES}
        observed={str(row.get("sample_id")) for row in summary.get("rows",[])}
        if expected != observed:
            raise ValueError("E3.13 recovery summary scope is not exact")
        latest.update({str(row["sample_id"]):row for row in summary.get("rows",[])})
    from collections import Counter
    terminal,routes=Counter(),Counter(); coverage={arm:set() for arm in ARMS}; rag_evidence_closed=False
    for sample_id,row in latest.items():
        routes[str(row.get("final_route") or "none")]+=1
        if row.get("delivery_status") in DELIVERY_FAILURES:
            terminal["delivery_shortfall"]+=1
        elif row.get("winner"):
            terminal["accepted"]+=1
            target=(by_id[sample_id].get("private") or {}).get("e313_route_coverage")
            if target == row.get("final_route"):
                coverage[by_id[sample_id]["arm"]].add(target)
            if row.get("final_route") == "rag":
                try:
                    trace=json.loads(Path(str(row.get("parent_path") or "")).read_text()).get("tool_trace",[])
                    rag_evidence_closed |= any(event.get("call",{}).get("name")=="agrinet_rag_search" and isinstance(event.get("response"),dict) for event in trace if isinstance(event,dict))
                except (OSError,json.JSONDecodeError):
                    pass
        else:
            terminal[str(row.get("quality_status") or "quality_rejected")]+=1
    coverage_winners={arm:sorted(values) for arm,values in coverage.items()}
    observed=any(row.get("delivery_status") == "delivered" for row in latest.values())
    gate=bool(observed and rag_evidence_closed and all(set(coverage_winners[arm]) == set(ROUTES) for arm in ARMS))
    return {"schema_version":"agrinet.e313-audit-final-report/v1","protocol":E313_PROTOCOL,"rows":32,"rounds":["R0","R1","R2"],"terminal_counts":dict(terminal),"final_routes":dict(routes),"coverage_winners":coverage_winners,"protocol_observed":observed,"rag_evidence_closed":rag_evidence_closed,"protocol_gate_passed":gate,"full_campaign_remaining_candidates":910,"full_campaign_expansion":"requires_separate_manifest" if gate else "not_authorized","training_eligible":False,"training_authorized":False,"sft_may_start":False}

def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "plan-replenishment":
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--summary",type=Path,required=True)
        parser.add_argument("--next-round",choices=("R1","R2"),required=True)
        parser.add_argument("--output",type=Path,required=True)
        args=parser.parse_args(argv[1:])
        manifest=write_e313_replenishment_manifest(summary=json.loads(args.summary.read_text()),next_round=args.next_round,output=args.output)
        print(json.dumps({"round":manifest["round"],"rows":len(manifest["work_items"])}))
        return 0
    if argv and argv[0] == "audit-final-report":
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--source",type=Path,required=True)
        parser.add_argument("--r0-summary",type=Path,required=True)
        parser.add_argument("--r1-summary",type=Path,required=True)
        parser.add_argument("--r2-summary",type=Path,required=True)
        parser.add_argument("--output",type=Path,required=True)
        args=parser.parse_args(argv[1:])
        if args.output.exists():
            raise ValueError("E3.13 audit report destination is immutable")
        source=[json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
        summaries=[json.loads(path.read_text()) for path in (args.r0_summary,args.r1_summary,args.r2_summary)]
        report=e313_layered_audit_report(source_rows=source,summaries=summaries)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"protocol_gate_passed":report["protocol_gate_passed"]}))
        return 0 if report["protocol_gate_passed"] else 2
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-source",type=Path,required=True)
    parser.add_argument("--prior",action="append",type=Path,required=True)
    parser.add_argument("--coverage-output",type=Path,required=True)
    parser.add_argument("--source-output",type=Path,required=True)
    parser.add_argument("--campaign-id",required=True)
    parser.add_argument("--manifest-output",type=Path,required=True)
    args=parser.parse_args(argv)
    if any(path.exists() for path in (args.coverage_output,args.source_output,args.manifest_output)):
        raise ValueError("E3.13 preparation destinations are immutable")
    candidates=[json.loads(line) for line in args.candidate_source.read_text().splitlines() if line.strip()]
    prior=[json.loads(line) for path in args.prior for line in path.read_text().splitlines() if line.strip()]
    rows=select_e311_audit(candidates,prior_rows=prior,seed="e313-audit-v1")
    coverage=select_e310_coverage(rows,seed="e313-coverage-v1")
    source=materialize_e313_source(rows,coverage)
    args.coverage_output.parent.mkdir(parents=True,exist_ok=True)
    args.coverage_output.write_text(json.dumps(coverage,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    args.source_output.parent.mkdir(parents=True,exist_ok=True)
    args.source_output.write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in source))
    manifest=write_e313_manifest(source=args.source_output,campaign_id=args.campaign_id,output=args.manifest_output)
    print(json.dumps({"rows":len(source),"prior_rows":len(prior),"coverage_rows":len(coverage),"manifest_rows":len(manifest["work_items"])}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
