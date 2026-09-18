"""Prepare the paired 28-image E3.27 visual-only Top-3 RAG preflight."""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
from typing import Any
from agrinet.rag.e324_structured_rag import digest,rows
from agrinet.rag.e326_structured_rag import CELL_TARGETS,FOLD_TARGETS,IDENTITIES
from agrinet.rag.e327_contract import PROTOCOL

SCHEMA="agrinet.e327-visual-top3-rag-manifest/v1"
ROWS=28; TOKEN_CAP=500_000; RESERVES={"closure":4500,"private_audit":3000}

def _write(path:Path,text:str)->None:
    if path.exists(): raise ValueError(f"E3.27 destination is immutable: {path}")
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)

def prepare(*,e326_source:Path,ablation_report:Path,ablation_audit:Path,output_root:Path)->dict[str,Any]:
    base=rows(e326_source); report=json.loads(ablation_report.read_text()); audit=json.loads(ablation_audit.read_text())
    if len(base)!=ROWS or any(r.get("e39_protocol")!="agrinet.e326-structured-rag/v1" for r in base): raise ValueError("E3.27 requires frozen E3.26 source")
    if report.get("protocol")!="agrinet.e326-semantic-retrieval-ablation/v1" or report.get("best_strategy")!="visual_only" or report.get("provider_requests")!=0: raise ValueError("E3.27 ablation report invalid")
    visual=report["strategy_metrics"]["visual_only"]
    if visual["all"]["recall_at_3"]!=0.75 or visual["open_pest"]["recall_at_3"]!=5/7: raise ValueError("E3.27 visual Top-3 evidence changed")
    if audit.get("ablation_audit_passed") is not True or audit.get("generated_query_truth_name_occurrences")!=0: raise ValueError("E3.27 ablation audit invalid")
    if Counter(f"{r['question_type']}/{r['task_domain']}" for r in base)!=Counter(CELL_TARGETS) or Counter(r["classifier"]["held_out_fold"] for r in base)!=Counter(FOLD_TARGETS): raise ValueError("E3.27 cohort coverage invalid")
    frozen=[]
    for old in sorted(base,key=lambda r:r["sample_id"]):
        private=dict(old["private"]); private.update({"e327_source_protocol":old["e39_protocol"],"e327_stratum":f"{old['question_type']}/{old['task_domain']}"})
        frozen.append({**old,"e39_protocol":PROTOCOL,"private":private,"training_eligible":False,"training_authorized":False,"sft_may_start":False})
    source_path=output_root/"source.jsonl"; manifest_path=output_root/"manifest-r0.json"
    _write(source_path,"".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in frozen))
    bindings={"e326_source":{"path":str(e326_source),"sha256":digest(e326_source)},"ablation_report":{"path":str(ablation_report),"sha256":digest(ablation_report)},"ablation_audit":{"path":str(ablation_audit),"sha256":digest(ablation_audit)}}
    controls={"uncached_input_token_cap":TOKEN_CAP,"transport_image_max_side":512,"request_timeout_seconds":180,"max_rag_searches":1,"rag_top_k":3,"rag_retrieval_type":"visual","reservation_uncached_tokens":RESERVES,"recovery_rounds":["R0","R1","R2"],"quality_repair_max":1}
    manifest={"schema_version":SCHEMA,"protocol":PROTOCOL,"campaign_id":"e327-visual-top3-rag-v1","round":"R0","source":str(source_path),"source_sha256":digest(source_path),"source_rows_expected":ROWS,"immutable_inputs":bindings,"identity_keys":list(IDENTITIES),"cell_targets":CELL_TARGETS,"classifier_fold_targets":FOLD_TARGETS,"paired_with":"e326-structured-rag-v1","work_items":[{"work_id":f"R0:{r['sample_id']}:e327","sample_id":r["sample_id"],"round":"R0","attempt_ordinal":0,"quality_attempt_ordinal":0,"resume_operation":"rag","predecessor_request_id":None} for r in frozen],"workers":4,"micu_intent_limit":8000,"pipeline":["frozen_classifier_card","fixed_visual_top3_rag","short_id_only_closure","deterministic_hcv_renderer","private_semantic_audit"],"forbidden_tools":["agrinet_reject"],"collection_controls":controls,"gate":{"minimum_semantic_correct":21,"minimum_per_cell_correct":5,"minimum_open_pest_correct":5},"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    _write(manifest_path,json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    return {"rows":ROWS,"cells":CELL_TARGETS,"folds":FOLD_TARGETS,"source":str(source_path),"manifest":str(manifest_path),"provider_requests":0,"sft_may_start":False}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--e326-source",type=Path,required=True); p.add_argument("--ablation-report",type=Path,required=True); p.add_argument("--ablation-audit",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args(argv)
    print(json.dumps(prepare(e326_source=a.e326_source,ablation_report=a.ablation_report,ablation_audit=a.ablation_audit,output_root=a.output_root),ensure_ascii=False,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
