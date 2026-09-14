"""Prepare the immutable E3.23 RAG-only preflight from E3.22 v3."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from typing import Any

from agrinet.rag.e319_rag_closure_audit import PROMPTS

PROTOCOL = "agrinet.e323-rag-preflight/v4"
SCHEMA = "agrinet.e323-rag-preflight-manifest/v4"
CAMPAIGN_ID = "e323-rag-preflight-v4"

def digest(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def _write_immutable(path: Path, text: str) -> None:
    if path.exists():
        raise ValueError(f"E3.23 destination is immutable: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def prepare(*, e322_source: Path, outcome_paths: list[Path], final_report: Path,
            gate_decision: Path, output_root: Path) -> dict[str, Any]:
    source_rows=read_jsonl(e322_source); by_id={r["sample_id"]:r for r in source_rows}
    if len(source_rows)!=32 or len(by_id)!=32:
        raise ValueError("E3.23 requires the frozen 32-row E3.22 v3 source")
    latest: dict[str, dict[str, Any]]={}
    bindings=[]
    for path in outcome_paths:
        document=json.loads(path.read_text(encoding="utf-8"))
        bindings.append({"path":str(path),"sha256":digest(path)})
        for outcome in document.get("outcomes") or []:
            latest[str(outcome["sample_id"])]=outcome
    selected=[sid for sid,outcome in latest.items() if outcome.get("disposition")=="future_rag"]
    if len(selected)!=16 or any(by_id[sid].get("arm")!="simulated_unknown" for sid in selected):
        raise ValueError("E3.23 requires exactly 16 simulated-Unknown future-RAG rows")
    frozen=[]
    groups={"autonomous_insufficient_evidence":0,"oracle_rescued_unsafe_accept":0}
    for sid in sorted(selected):
        prior=latest[sid]
        group=("autonomous_insufficient_evidence" if prior.get("autonomous_unknown_deferral")
               else "oracle_rescued_unsafe_accept")
        groups[group]+=1
        row=dict(by_id[sid]); private=dict(row.get("private") or {})
        private["e323_prior_group"]=group
        private["e322_terminal_outcome_sha256"]=hashlib.sha256(json.dumps(prior,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        row.update({"e39_protocol":PROTOCOL,"teacher_system_prompts":dict(PROMPTS),"private":private,
                    "training_eligible":False,"training_authorized":False,"sft_may_start":False})
        frozen.append(row)
    if groups!={"autonomous_insufficient_evidence":6,"oracle_rescued_unsafe_accept":10}:
        raise ValueError(f"E3.23 private group split changed: {groups}")
    source_path=output_root/"source.jsonl"; manifest_path=output_root/"manifest-r0.json"
    source_text="".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in frozen)
    _write_immutable(source_path,source_text)
    immutable={"e322_source":{"path":str(e322_source),"sha256":digest(e322_source)},
               "e322_outcomes":bindings,
               "e322_final_report":{"path":str(final_report),"sha256":digest(final_report)},
               "e322_gate_decision":{"path":str(gate_decision),"sha256":digest(gate_decision)}}
    manifest={"schema_version":SCHEMA,"protocol":PROTOCOL,"campaign_id":CAMPAIGN_ID,"round":"R0",
      "source":str(source_path),"source_sha256":digest(source_path),"source_rows_expected":16,
      "immutable_inputs":immutable,"private_group_counts":groups,"rag_only":True,
      "work_items":[{"work_id":f"R0:{r['sample_id']}:e323-rag","sample_id":r["sample_id"],"round":"R0",
        "resume_route":"rag","resume_operation":"generation","attempt_ordinal":0,"quality_attempt_ordinal":0,
        "predecessor_request_id":None,"prompt_revision":"base"} for r in frozen],
      "workers":4,"micu_intent_limit":8000,"tools":["agrinet_classifier_predict","agrinet_classifier_expand","agrinet_rag_search"],
      "forbidden_tools":["agrinet_reject"],
      "collection_controls":{"uncached_input_token_cap":300000,"transport_image_max_side":512,"request_timeout_seconds":180,
        "max_public_turns_per_route":12,"max_rag_searches":3,"reservation_uncached_tokens":{"generation":12000,"private_audit":5000},
        "recovery_rounds":["R0","R1","R2"],"quality_repair_max":1},
      "training_eligible":False,"training_authorized":False,"sft_may_start":False}
    _write_immutable(manifest_path,json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    return {"rows":16,"groups":groups,"source":str(source_path),"manifest":str(manifest_path),"provider_requests":0,"sft_may_start":False}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--e322-source",type=Path,required=True); p.add_argument("--e322-outcome",type=Path,action="append",required=True)
    p.add_argument("--e322-final-report",type=Path,required=True); p.add_argument("--e322-gate-decision",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args(argv)
    print(json.dumps(prepare(e322_source=a.e322_source,outcome_paths=a.e322_outcome,final_report=a.e322_final_report,gate_decision=a.e322_gate_decision,output_root=a.output_root),ensure_ascii=False,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
