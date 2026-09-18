"""Prepare the immutable E3.28 full Classifier campaign."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from typing import Any
from agrinet.rag.e322_presample import IDENTITIES, classifier_prompt, digest, fold_of, rows

PROTOCOL = "agrinet.e328-classifier-full/v1"
SCHEMA = "agrinet.e328-classifier-full-manifest/v1"
ROWS = 525
REUSED = 32
NEW = 493
SHARD_SIZE = 64
FLAGS = {"training_eligible": False, "training_authorized": False, "sft_may_start": False}
ROUND_FILES = ("r0.json", "q1.json", "r1.json", "q1-r1.json", "r2.json", "q1-r2.json")

def _write_json(path: Path, value: Any) -> None:
    text=json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n"
    if path.exists(): raise ValueError(f"E3.28 prepare output is immutable: {path}")
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)

def _latest(campaign: Path) -> dict[str,dict[str,Any]]:
    latest={}
    for name in ROUND_FILES:
        path=campaign/"outcomes"/name
        if path.is_file():
            for out in json.loads(path.read_text()).get("outcomes") or []: latest[out["sample_id"]]=out
    return latest

def prepare(*, queue_path:Path, direct_source:Path, e320_source:Path, e322_source:Path,
            e322_campaign:Path, e322_report:Path, e322_audit:Path, e322_gate:Path,
            output_root:Path, protocol:str=PROTOCOL, campaign_id:str="e328-classifier-full-v1",
            provenance_new:str="new_e328", provenance_reused:str="reused_e322_v3",
            provenance_key:str="e328_provenance", schema:str=SCHEMA,
            shard_schema:str="agrinet.e328-classifier-full-shard-manifest/v1",
            prepare_audit_schema:str="agrinet.e328-classifier-full-prepare-audit/v1",
            terminal_unknown_delivery_max_fraction:float|None=None,
            shards_are_scheduling_only:bool=False) -> dict[str,Any]:
    targets=[output_root/"source.jsonl",output_root/"manifest.json",output_root/"prepare-audit.json"]
    if any(p.exists() for p in targets): raise ValueError("E3.28 prepare outputs are immutable")
    queue=json.loads(queue_path.read_text()); q=queue.get("queue") or []
    if queue.get("count")!=ROWS or len(q)!=ROWS or len({x["sample_id"] for x in q})!=ROWS: raise ValueError("E3.28 frozen queue must contain 525 unique rows")
    source_by={r["sample_id"]:r for r in rows(direct_source)+rows(e320_source)}
    ids={x["sample_id"] for x in q}
    if not ids.issubset(source_by): raise ValueError("E3.28 source binding incomplete")
    reused_rows=rows(e322_source); reused_ids={r["sample_id"] for r in reused_rows}
    latest=_latest(e322_campaign); gate=json.loads(e322_gate.read_text())
    if len(reused_ids)!=REUSED or set(latest)!=reused_ids or gate.get("presample_gate_passed") is not True: raise ValueError("E3.28 E3.22 v3 reuse gate invalid")
    if any(latest[s].get("disposition") not in {"semantic_correct","future_rag"} for s in reused_ids): raise ValueError("E3.28 reused terminal outcome invalid")
    ordered=[]
    for sid in sorted(ids):
        r=source_by[sid]; ordered.append({**r,"e39_protocol":protocol,provenance_key:provenance_reused if sid in reused_ids else provenance_new,"teacher_system_prompts":{"classifier":classifier_prompt(r)},**FLAGS})
    if len(ordered)!=ROWS or sum(r[provenance_key]==provenance_new for r in ordered)!=NEW: raise ValueError("Classifier 32+493 conservation failed")
    expected={"arms":{"known":274,"simulated_unknown":251},"questions":{"open":470,"option":55},"domains":{"disease":365,"pest":160}}
    actual={"arms":dict(Counter(r["arm"] for r in ordered)),"questions":dict(Counter(r["question_type"] for r in ordered)),"domains":dict(Counter(r["domain"] for r in ordered))}
    if actual!=expected or len({r["canonical_class_code"] for r in ordered})!=105: raise ValueError(f"E3.28 frozen source distribution changed: {actual}")
    for key in IDENTITIES:
        if len({r[key] for r in ordered})!=ROWS: raise ValueError(f"E3.28 identity is not unique: {key}")
    if len({r["classifier"]["checkpoint_sha256"] for r in ordered})!=6: raise ValueError("E3.28 six-checkpoint coverage failed")
    output_root.mkdir(parents=True,exist_ok=True)
    targets[0].write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in ordered))
    new_rows=[r for r in ordered if r[provenance_key]==provenance_new]
    bindings={name:{"path":str(path),"sha256":digest(path)} for name,path in {"queue":queue_path,"direct_source":direct_source,"e320_source":e320_source,"e322_source":e322_source,"e322_final_report":e322_report,"e322_artifact_audit":e322_audit,"e322_gate":e322_gate}.items()}
    for name in ROUND_FILES:
        path=e322_campaign/"outcomes"/name
        if path.is_file(): bindings[f"e322_outcome_{name[:-5]}"]={"path":str(path),"sha256":digest(path)}
    if terminal_unknown_delivery_max_fraction is not None and not 0.0 < terminal_unknown_delivery_max_fraction <= 1.0:
        raise ValueError("terminal unknown-delivery tolerance must be in (0, 1]")
    controls={"uncached_input_token_cap":4000000,"global_micu_intent_limit":8000,"transport_image_max_side":512,"request_timeout_seconds":180,"reservation_uncached_tokens":{"generation":8000,"private_audit":5000},"recovery_rounds":["R0","R1","R2"],"quality_repair_max":1,"terminal_unknown_delivery_max_fraction":terminal_unknown_delivery_max_fraction}
    shard_refs=[]
    for index,start in enumerate(range(0,len(new_rows),SHARD_SIZE)):
        batch=new_rows[start:start+SHARD_SIZE]; path=output_root/"manifests"/f"shard-{index:02d}.json"
        value={"schema_version":shard_schema,"protocol":protocol,"shard_index":index,"source_sha256":digest(targets[0]),"rows":len(batch, ),"sample_ids":[r["sample_id"] for r in batch],"work_items":[{"work_id":f"R0:{r['sample_id']}:{campaign_id}","sample_id":r["sample_id"],"round":"R0","resume_route":"classifier","resume_operation":"generation","attempt_ordinal":0,"quality_attempt_ordinal":0,"predecessor_request_id":None,"prompt_revision":"base"} for r in batch],"workers":4,**FLAGS}
        _write_json(path,value); shard_refs.append({"path":str(path),"sha256":digest(path),"rows":len(batch),"shard_index":index})
    if [x["rows"] for x in shard_refs]!=[64]*7+[45]: raise ValueError("E3.28 deterministic shard sizes invalid")
    manifest={"schema_version":schema,"protocol":protocol,"campaign_id":campaign_id,"source":str(targets[0]),"source_sha256":digest(targets[0]),"source_rows_expected":ROWS,"reused_rows":REUSED,"new_rows":NEW,"immutable_inputs":bindings,"shards":shard_refs,"workers_per_shard":4,"orchestrator":"serial_immutable_shards","shards_are_scheduling_only":shards_are_scheduling_only,"shared_collection_controls":controls,"tools":["agrinet_classifier_predict","agrinet_classifier_expand"],"forbidden_tools":["agrinet_rag_search","agrinet_reject"],**FLAGS}
    _write_json(targets[1],manifest)
    audit={"schema_version":prepare_audit_schema,"protocol":protocol,"source_sha256":digest(targets[0]),"manifest_sha256":digest(targets[1]),"rows":ROWS,"reused_rows":REUSED,"new_rows":NEW,"shard_rows":[x["rows"] for x in shard_refs],"identity_unique":{k:len({r[k] for r in ordered}) for k in IDENTITIES},"truth_classes":len({r["canonical_class_code"] for r in ordered}),"distributions":actual,"folds":dict(Counter(f"{r['arm']}:{fold_of(r)}" for r in ordered)),"checkpoint_sha256":sorted({r["classifier"]["checkpoint_sha256"] for r in ordered}),"provider_intents_created":0,**FLAGS}
    _write_json(targets[2],audit); return audit

def main(argv=None):
    p=argparse.ArgumentParser()
    for name in ("queue","direct-source","e320-source","e322-source","e322-campaign","e322-report","e322-audit","e322-gate","output-root"): p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args(argv); print(json.dumps(prepare(queue_path=a.queue,direct_source=a.direct_source,e320_source=a.e320_source,e322_source=a.e322_source,e322_campaign=a.e322_campaign,e322_report=a.e322_report,e322_audit=a.e322_audit,e322_gate=a.e322_gate,output_root=a.output_root),sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
