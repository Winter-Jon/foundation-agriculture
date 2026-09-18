"""Derive immutable dynamic E3.29 RAG source from a passed E3.28 campaign."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from typing import Any
from agrinet.rag.e322_presample import digest, rows
from agrinet.rag.e328_classifier_full import FLAGS

PROTOCOL="agrinet.e329-visual-top3-rag-full/v1"
SCHEMA="agrinet.e329-visual-top3-rag-full-manifest/v1"
SHARD_SIZE=64

def _write(path:Path,value:Any)->None:
    text=json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n"
    if path.exists(): raise ValueError(f"E3.29 immutable output exists: {path}")
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)

def prepare(*,e328_source:Path,e328_report:Path,e328_audit:Path,e328_gate:Path,e327_source:Path,output_root:Path,allow_partial_classifier:bool=False)->dict[str,Any]:
    targets=(output_root/"source.jsonl",output_root/"manifest.json",output_root/"prepare-audit.json")
    if any(x.exists() for x in targets): raise ValueError("E3.29 prepare outputs are immutable")
    gate=json.loads(e328_gate.read_text()); report=json.loads(e328_report.read_text()); audit=json.loads(e328_audit.read_text())
    full_gate=gate.get("classifier_gate_passed") is True and audit.get("artifact_audit_passed") and report.get("campaign_gate_candidate")
    partial_gate=(allow_partial_classifier and gate.get("partial_rag_input_gate_passed") is True and report.get("collection_complete") is True
                  and report.get("rag_input_gate_candidate") is True
                  and audit.get("artifact_audit_passed"))
    if not (full_gate or partial_gate): raise ValueError("E3.29 requires a passed classifier gate or an audited partial RAG input gate")
    source_rows=rows(e328_source); by={r["sample_id"]:r for r in source_rows}; future=report.get("future_rag_queue") or []
    ids=[x.get("sample_id") for x in future]
    if len(ids)!=len(set(ids)) or any(s not in by for s in ids): raise ValueError("E3.29 future_rag queue binding invalid")
    old={r["sample_id"] for r in rows(e327_source)}
    frozen=[]
    for sid in sorted(ids):
        row=by[sid]; frozen.append({**row,"e39_protocol":PROTOCOL,"e329_classifier_outcome":next(x for x in future if x["sample_id"]==sid),"e327_overlap":sid in old,**FLAGS})
    output_root.mkdir(parents=True,exist_ok=True); targets[0].write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in frozen))
    bindings={name:{"path":str(path),"sha256":digest(path)} for name,path in {"e328_source":e328_source,"e328_report":e328_report,"e328_audit":e328_audit,"e328_gate":e328_gate,"e327_source":e327_source}.items()}
    shards=[]
    for index,start in enumerate(range(0,len(frozen),SHARD_SIZE)):
        batch=frozen[start:start+SHARD_SIZE]; path=output_root/"manifests"/f"shard-{index:02d}.json"
        value={"schema_version":"agrinet.e329-visual-top3-rag-full-shard-manifest/v1","protocol":PROTOCOL,"shard_index":index,"rows":len(batch),"source_sha256":digest(targets[0]),"sample_ids":[r["sample_id"] for r in batch],"work_items":[{"work_id":f"R0:{r['sample_id']}:e329-visual-top3","sample_id":r["sample_id"],"round":"R0","attempt_ordinal":0,"quality_attempt_ordinal":0,"resume_operation":"rag","predecessor_request_id":None} for r in batch],"workers":4,**FLAGS}
        _write(path,value); shards.append({"path":str(path),"sha256":digest(path),"rows":len(batch),"shard_index":index})
    controls={"uncached_input_token_cap":2000000,"global_micu_intent_limit":8000,"transport_image_max_side":512,"request_timeout_seconds":180,"max_rag_searches":1,"rag_top_k":3,"rag_retrieval_type":"visual","reservation_uncached_tokens":{"closure":4500,"private_audit":3000},"recovery_rounds":["R0","R1","R2"],"quality_repair_max":1}
    manifest={"schema_version":SCHEMA,"protocol":PROTOCOL,"campaign_id":"e329-visual-top3-rag-full-v1","source":str(targets[0]),"source_sha256":digest(targets[0]),"source_rows_expected":len(frozen),"dynamic_future_rag_count":len(frozen),"classifier_collection_scope":"partial_terminal_safe_subset" if partial_gate else "full_gate","classifier_terminal_shortfall_count":len(report.get("classifier_terminal_shortfalls") or []),"immutable_inputs":bindings,"shards":shards,"workers_per_shard":4,"orchestrator":"serial_immutable_shards","pipeline":["fixed_visual_top3_rag","visibility_grounded_closure","private_semantic_audit"],"forbidden_tools":["agrinet_reject","planner","ranker"],"collection_controls":controls,**FLAGS}
    _write(targets[1],manifest)
    audit={"schema_version":"agrinet.e329-visual-top3-rag-full-prepare-audit/v1","protocol":PROTOCOL,"source_sha256":digest(targets[0]),"manifest_sha256":digest(targets[1]),"rows":len(frozen),"shard_rows":[x["rows"] for x in shards],"e327_overlap":sum(r["e327_overlap"] for r in frozen),"e327_non_overlap":sum(not r["e327_overlap"] for r in frozen),"source_dispositions":dict(Counter(x.get("reason") for x in future)),"provider_intents_created":0,**FLAGS}
    _write(targets[2],audit); return audit

def main(argv=None):
    p=argparse.ArgumentParser()
    for name in ("e328-source","e328-report","e328-audit","e328-gate","e327-source","output-root"): p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args(argv); print(json.dumps(prepare(e328_source=a.e328_source,e328_report=a.e328_report,e328_audit=a.e328_audit,e328_gate=a.e328_gate,e327_source=a.e327_source,output_root=a.output_root),sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
