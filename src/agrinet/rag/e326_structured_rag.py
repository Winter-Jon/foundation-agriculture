"""Prepare the 28-image morphology-first E3.26 structured RAG preflight."""
from __future__ import annotations
import argparse,hashlib,itertools,json
from collections import Counter
from pathlib import Path
from typing import Any
from agrinet.rag.e324_structured_rag import digest,rows

PROTOCOL="agrinet.e326-structured-rag/v1"
SCHEMA="agrinet.e326-structured-rag-manifest/v1"
IDENTITIES=("sample_id","image_sha256","source_group_id","near_duplicate_group_id")
CELLS=(("open","disease"),("open","pest"),("option","disease"),("option","pest"))
CELL_TARGETS={f"{q}/{d}":7 for q,d in CELLS}; FOLD_TARGETS={0:9,1:10,2:9}
ROWS=28; TOKEN_CAP=900_000; RESERVES={"planner":1500,"closure":6000,"private_audit":4000}

def _cell(row:dict[str,Any])->str: return f"{row['question_type']}/{row['task_domain']}"
def _write(path:Path,text:str)->None:
    if path.exists(): raise ValueError(f"E3.26 destination is immutable: {path}")
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)

def select(source:list[dict[str,Any]],excluded:list[dict[str,Any]],seed:str="e326-structured-rag-v1")->list[dict[str,Any]]:
    blocked={key:{r[key] for r in excluded} for key in IDENTITIES}
    eligible=[r for r in source if all(r[key] not in blocked[key] for key in IDENTITIES)]
    if len(eligible)!=31: raise ValueError(f"E3.26 expected 31 identity-disjoint E3.18 candidates, got {len(eligible)}")
    surplus=[[r for r in eligible if _cell(r)==cell] for cell,count in Counter(_cell(r) for r in eligible).items() if count==8]
    candidates=[]
    for removed in itertools.product(*surplus):
        chosen=[r for r in eligible if r not in removed]
        if len(chosen)!=ROWS or len({r["canonical_class_code"] for r in chosen})!=ROWS: continue
        if Counter(_cell(r) for r in chosen)!=Counter(CELL_TARGETS): continue
        if Counter(r["classifier"]["held_out_fold"] for r in chosen)!=Counter(FOLD_TARGETS): continue
        tie=hashlib.sha256((seed+":"+":".join(r["sample_id"] for r in removed)).encode()).hexdigest()
        candidates.append((tie,chosen))
    if not candidates: raise ValueError("E3.26 cannot satisfy cohort constraints")
    return min(candidates,key=lambda x:x[0])[1]

def prepare(*,e318_source:Path,excluded_sources:list[Path],output_root:Path)->dict[str,Any]:
    base=rows(e318_source)
    if len(base)!=32 or any(r.get("e39_protocol")!="agrinet.e318-all-unknown-512-rag-audit/v1" for r in base): raise ValueError("E3.26 requires frozen E3.18 RAG-witness source")
    excluded=[r for path in excluded_sources for r in rows(path)]; chosen=select(base,excluded); frozen=[]
    for old in sorted(chosen,key=lambda r:r["sample_id"]):
        private=dict(old.get("private") or {})
        if not (private.get("audit_protocol") or {}).get("rag_witness"): raise ValueError("E3.26 candidate lacks RAG eligibility")
        private.update({"e326_source_protocol":old["e39_protocol"],"e326_stratum":_cell(old)})
        frozen.append({**old,"e39_protocol":PROTOCOL,"private":private,"training_eligible":False,"training_authorized":False,"sft_may_start":False})
    source_path=output_root/"source.jsonl"; manifest_path=output_root/"manifest-r0.json"
    _write(source_path,"".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in frozen))
    bindings={"e318_source":{"path":str(e318_source),"sha256":digest(e318_source)}}
    bindings.update({f"excluded_source_{i}":{"path":str(path),"sha256":digest(path)} for i,path in enumerate(excluded_sources,1)})
    controls={"uncached_input_token_cap":TOKEN_CAP,"transport_image_max_side":512,"request_timeout_seconds":180,"max_rag_searches":1,"rag_top_k":8,"rag_retrieval_type":"balanced","rag_ranker":"rrf","reservation_uncached_tokens":RESERVES,"recovery_rounds":["R0","R1","R2"],"quality_repair_max":1}
    manifest={"schema_version":SCHEMA,"protocol":PROTOCOL,"campaign_id":"e326-structured-rag-v1","round":"R0","source":str(source_path),"source_sha256":digest(source_path),"source_rows_expected":ROWS,"immutable_inputs":bindings,"identity_keys":list(IDENTITIES),"cell_targets":CELL_TARGETS,"classifier_fold_targets":FOLD_TARGETS,"work_items":[{"work_id":f"R0:{r['sample_id']}:e326","sample_id":r["sample_id"],"round":"R0","attempt_ordinal":0,"quality_attempt_ordinal":0,"resume_operation":"planner","predecessor_request_id":None} for r in frozen],"workers":4,"micu_intent_limit":8000,"pipeline":["frozen_classifier_card","morphology_first_planner","single_balanced_top8_rag","id_only_closure","deterministic_hcv_renderer","private_semantic_audit"],"forbidden_tools":["agrinet_reject"],"collection_controls":controls,"gate":{"minimum_semantic_correct":23,"minimum_per_cell_correct":5,"minimum_open_pest_correct":5},"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    _write(manifest_path,json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    return {"rows":ROWS,"cells":CELL_TARGETS,"folds":FOLD_TARGETS,"source":str(source_path),"manifest":str(manifest_path),"provider_requests":0,"sft_may_start":False}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--e318-source",type=Path,required=True); p.add_argument("--excluded-source",type=Path,action="append",required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args(argv)
    print(json.dumps(prepare(e318_source=a.e318_source,excluded_sources=a.excluded_source,output_root=a.output_root),ensure_ascii=False,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
