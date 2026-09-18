"""E3.24 structured evidence-to-answer RAG preflight preparation."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from typing import Any

PROTOCOL="agrinet.e324-structured-rag/v2"
SCHEMA="agrinet.e324-structured-rag-manifest/v2"
GROUPS={"autonomous_insufficient_evidence":4,"oracle_rescued_unsafe_accept":4}
IDENTITIES=("sample_id","image_sha256","source_group_id","near_duplicate_group_id")

def digest(path:Path)->str: return hashlib.file_digest(path.open("rb"),"sha256").hexdigest()
def rows(path:Path)->list[dict[str,Any]]: return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def _write(path:Path,text:str)->None:
    if path.exists(): raise ValueError(f"E3.24 destination is immutable: {path}")
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)

def prepare(*,e323_source:Path,e323_report:Path,e323_gate:Path,output_root:Path,seed:str="e324-structured-rag-v1")->dict[str,Any]:
    source=rows(e323_source)
    if len(source)!=16 or any(r.get("e39_protocol")!="agrinet.e323-rag-preflight/v4" for r in source): raise ValueError("E3.24 requires frozen E3.23 v4 source")
    chosen=[]
    for group,count in GROUPS.items():
        pool=[r for r in source if (r.get("private") or {}).get("e323_prior_group")==group]
        pool.sort(key=lambda r:hashlib.sha256(f"{seed}:{r['sample_id']}".encode()).hexdigest())
        if len(pool)<count: raise ValueError(f"E3.24 lacks private stratum {group}")
        chosen.extend(pool[:count])
    if any(len({r[k] for r in chosen})!=8 for k in IDENTITIES) or len({r["canonical_class_code"] for r in chosen})!=8: raise ValueError("E3.24 identity or truth-class collision")
    frozen=[]
    for old in sorted(chosen,key=lambda r:r["sample_id"]):
        r=dict(old); private=dict(r.get("private") or {}); private["e324_stratum"]=private.pop("e323_prior_group")
        r.update({"e39_protocol":PROTOCOL,"private":private,"training_eligible":False,"training_authorized":False,"sft_may_start":False}); frozen.append(r)
    source_path=output_root/"source.jsonl"; manifest_path=output_root/"manifest-r0.json"
    _write(source_path,"".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in frozen))
    bindings={"e323_source":{"path":str(e323_source),"sha256":digest(e323_source)},"e323_report":{"path":str(e323_report),"sha256":digest(e323_report)},"e323_gate":{"path":str(e323_gate),"sha256":digest(e323_gate)}}
    manifest={"schema_version":SCHEMA,"protocol":PROTOCOL,"campaign_id":"e324-structured-rag-v2","round":"R0","source":str(source_path),"source_sha256":digest(source_path),"source_rows_expected":8,"immutable_inputs":bindings,"private_group_counts":GROUPS,"work_items":[{"work_id":f"R0:{r['sample_id']}:e324","sample_id":r["sample_id"],"round":"R0","attempt_ordinal":0,"quality_attempt_ordinal":0,"resume_operation":"planner","predecessor_request_id":None} for r in frozen],"workers":4,"micu_intent_limit":8000,"pipeline":["frozen_classifier_card","structured_planner","local_rag_search","structured_closure","deterministic_hcv_renderer","private_semantic_audit"],"forbidden_tools":["agrinet_reject"],"collection_controls":{"uncached_input_token_cap":180000,"transport_image_max_side":512,"request_timeout_seconds":180,"max_rag_searches":1,"reservation_uncached_tokens":{"planner":1000,"closure":3500,"private_audit":4000},"recovery_rounds":["R0","R1","R2"],"quality_repair_max":1},"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    _write(manifest_path,json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    return {"rows":8,"groups":GROUPS,"source":str(source_path),"manifest":str(manifest_path),"provider_requests":0,"sft_may_start":False}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--e323-source",type=Path,required=True); p.add_argument("--e323-report",type=Path,required=True); p.add_argument("--e323-gate",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args(argv)
    print(json.dumps(prepare(e323_source=a.e323_source,e323_report=a.e323_report,e323_gate=a.e323_gate,output_root=a.output_root),ensure_ascii=False,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
