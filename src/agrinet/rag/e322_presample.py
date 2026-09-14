"""E3.22 immutable 32-image Classifier-only presample preparation."""
from __future__ import annotations

import argparse, hashlib, json
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.rag.e315_option_format_repair import CLASSIFIER_OPTION_PROMPT
from agrinet.rag.e313_hcv_cascade import PROMPTS as E313_PROMPTS

LEGACY_PROTOCOL = "agrinet.e322-classifier-presample/v1"
PREVIOUS_PROTOCOL = "agrinet.e322-classifier-presample/v2"
PROTOCOL = "agrinet.e322-classifier-presample/v3"
SCHEMA = "agrinet.e322-classifier-presample-manifest/v3"
SEED = "e322-classifier-presample-v1"
CAMPAIGN_ID = "e322-classifier-presample-v3"
RARE = {"e35-simulated_unknown-N05004-005832e4e6bcccd4d546", "e35-simulated_unknown-N05054-004d71407b0ecca3e34e"}
CELL_QUOTA = {("known","open","disease"):10,("known","open","pest"):5,("known","option","disease"):1,("known","option","pest"):0,("simulated_unknown","open","disease"):8,("simulated_unknown","open","pest"):4,("simulated_unknown","option","disease"):2,("simulated_unknown","option","pest"):2}
FOLD_QUOTA = {("known",0):5,("known",1):5,("known",2):6,("simulated_unknown",0):5,("simulated_unknown",1):6,("simulated_unknown",2):5}
IDENTITIES=("sample_id","image_sha256","source_group_id","near_duplicate_group_id")
OPEN_CLASSIFIER_PROMPT=(
    E313_PROMPTS["classifier"]
    + " For Open, the final <answer> must contain exactly one canonical public class name "
      "with no option letter, em dash, label prefix, or added explanation."
      " If no frozen-card candidate is adequately supported by visible evidence, output exactly <answer>INSUFFICIENT_EVIDENCE</answer>."
)
OPTION_CLASSIFIER_PROMPT=(CLASSIFIER_OPTION_PROMPT
    + " If no public option is adequately supported by visible evidence, output exactly <answer>INSUFFICIENT_EVIDENCE</answer>.")

def classifier_prompt(row:dict[str,Any])->str:
    question_type=str(row.get("question_type") or "")
    if question_type=="open": return OPEN_CLASSIFIER_PROMPT
    if question_type=="option": return OPTION_CLASSIFIER_PROMPT
    raise ValueError("E3.22 question type is invalid")

def digest(path:Path)->str: return hashlib.file_digest(path.open("rb"),"sha256").hexdigest()
def rows(path:Path)->list[dict[str,Any]]: return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def fold_of(row): return int(row["fold"] if row["arm"]=="known" else row["e3_fold"])

def select(queue:dict[str,Any], sources:list[dict[str,Any]])->list[dict[str,Any]]:
    q=queue.get("queue") or []
    if queue.get("count")!=525 or len(q)!=525 or len({x.get("sample_id") for x in q})!=525: raise ValueError("E3.22 requires frozen 525-row queue")
    by={str(r.get("sample_id")):r for r in sources}; ids={str(x["sample_id"]) for x in q}
    if not ids.issubset(by): raise ValueError("E3.22 queue source binding incomplete")
    pool=[by[x] for x in ids]; cq=dict(CELL_QUOTA); fq=dict(FOLD_QUOTA); chosen=[]; classes=set(); used={k:set() for k in IDENTITIES}
    def add(r):
        cell=(r.get("arm"),r.get("question_type"),r.get("domain")); fk=(r.get("arm"),fold_of(r)); code=str(r.get("canonical_class_code"))
        if cq.get(cell,0)<=0 or fq.get(fk,0)<=0 or code in classes or any(not isinstance(r.get(k),str) or r[k] in used[k] for k in IDENTITIES): return False
        card=r.get("classifier") or {}; top=card.get("top5")
        if not isinstance(top,list) or len(top)!=5 or len({card.get(x) for x in ("checkpoint_sha256","label_map_sha256","training_manifest_sha256","registry_sha256")})<4: raise ValueError("E3.22 classifier binding malformed")
        if r["arm"]=="simulated_unknown" and code in set(card.get("label_map_codes") or []): raise ValueError("E3.22 simulated-Unknown truth leakage")
        cq[cell]-=1; fq[fk]-=1; classes.add(code); chosen.append(r)
        for k in IDENTITIES: used[k].add(r[k])
        return True
    for sid in sorted(RARE):
        if sid not in by or not add(by[sid]): raise ValueError("E3.22 rare Option-pest binding failed")
    for r in sorted(pool,key=lambda x:hashlib.sha256(f"{SEED}:{x['sample_id']}".encode()).hexdigest()): add(r)
    if len(chosen)!=32 or any(cq.values()) or any(fq.values()) or len(classes)!=32: raise ValueError("E3.22 exact quota selection failed")
    if len({r["classifier"]["checkpoint_sha256"] for r in chosen})!=6: raise ValueError("E3.22 six checkpoint coverage failed")
    return chosen

def prepare(*,queue_path:Path,direct_source:Path,e320_source:Path,output_root:Path,checkpoint_paths:list[Path],label_maps:list[Path],training_manifests:list[Path],registry_path:Path,reference_v1_source:Path|None=None)->dict[str,Any]:
    source=output_root/"source.jsonl"; manifest=output_root/"manifest-r0.json"; audit=output_root/"prepare-audit.json"
    if any(p.exists() for p in (source,manifest,audit)): raise ValueError("E3.22 prepare outputs are immutable")
    if not (len(checkpoint_paths)==len(label_maps)==len(training_manifests)==6): raise ValueError("E3.22 requires six artifact bindings")
    queue=json.loads(queue_path.read_text()); selected=select(queue,rows(direct_source)+rows(e320_source))
    if reference_v1_source is None or not reference_v1_source.is_file(): raise ValueError("E3.22 v2 requires the frozen v1 reference source")
    reference=rows(reference_v1_source)
    if len(reference)!=32 or any({r[k] for r in selected}!={r[k] for r in reference} for k in IDENTITIES):
        raise ValueError("E3.22 v2 must be a paired repair on the exact v1 cohort")
    expected_cp={r["classifier"]["checkpoint_sha256"] for r in selected}
    if {digest(p) for p in checkpoint_paths}!=expected_cp: raise ValueError("E3.22 checkpoint entity hash mismatch")
    if {digest(p) for p in label_maps}!={r["classifier"]["label_map_sha256"] for r in selected}: raise ValueError("E3.22 label-map hash mismatch")
    if {digest(p) for p in training_manifests}!={r["classifier"]["training_manifest_sha256"] for r in selected}: raise ValueError("E3.22 training-manifest hash mismatch")
    registry_hashes={r["classifier"]["registry_sha256"] for r in selected}
    if len(registry_hashes)!=1: raise ValueError("E3.22 registry binding mismatch")
    if digest(registry_path)!=next(iter(registry_hashes)):
        raise ValueError("E3.22 canonical registry entity hash mismatch")
    frozen=[]
    for r in selected:
        x={**r,"e39_protocol":PROTOCOL,"teacher_system_prompts":{"classifier":classifier_prompt(r)},"training_eligible":False,"training_authorized":False,"sft_may_start":False}; frozen.append(x)
    output_root.mkdir(parents=True,exist_ok=True); source.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in frozen),encoding="utf-8")
    bindings={"queue":{"path":str(queue_path),"sha256":digest(queue_path)},"direct_source":{"path":str(direct_source),"sha256":digest(direct_source)},"e320_source":{"path":str(e320_source),"sha256":digest(e320_source)},"reference_v1_source":{"path":str(reference_v1_source),"sha256":digest(reference_v1_source)},"registry":{"path":str(registry_path),"entity_sha256":next(iter(registry_hashes)),"file_sha256":digest(registry_path)},"checkpoints":[{"path":str(p),"sha256":digest(p)} for p in checkpoint_paths],"label_maps":[{"path":str(p),"sha256":digest(p)} for p in label_maps],"training_manifests":[{"path":str(p),"sha256":digest(p)} for p in training_manifests]}
    payload={"schema_version":SCHEMA,"protocol":PROTOCOL,"campaign_id":CAMPAIGN_ID,"round":"R0","source":str(source),"source_sha256":digest(source),"source_rows_expected":32,"immutable_inputs":bindings,"sampling_relation":"paired_repair_reuses_exact_v1_cohort","live_collection_requires_explicit_authorization":True,"classifier_only":True,"work_items":[{"work_id":f"R0:{r['sample_id']}:e322-classifier-v3","sample_id":r["sample_id"],"round":"R0","resume_route":"classifier","resume_operation":"generation","attempt_ordinal":0,"quality_attempt_ordinal":0,"predecessor_request_id":None,"prompt_revision":"base"} for r in frozen],"workers":4,"micu_intent_limit":8000,"tools":["agrinet_classifier_predict","agrinet_classifier_expand"],"forbidden_tools":["agrinet_rag_search"],"collection_controls":{"uncached_input_token_cap":300000,"transport_image_max_side":512,"request_timeout_seconds":180,"reservation_uncached_tokens":{"generation":8000,"private_audit":5000},"recovery_rounds":["R0","R1","R2"],"quality_repair_max":1},"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    manifest.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    report={"schema_version":"agrinet.e322-prepare-audit/v1","source_sha256":digest(source),"manifest_sha256":digest(manifest),"rows":32,"classes":32,"cells":dict(Counter(f"{r['arm']}:{r['question_type']}:{r['domain']}" for r in frozen)),"folds":dict(Counter(f"{r['arm']}:{fold_of(r)}" for r in frozen)),"checkpoint_sha256":sorted(expected_cp),"rare_included":sorted(RARE),"identity_unique":{k:len({r[k] for r in frozen})==32 for k in IDENTITIES},"training_eligible":False,"training_authorized":False,"sft_may_start":False}; audit.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n"); return report

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--queue",type=Path,required=True); p.add_argument("--direct-source",type=Path,required=True); p.add_argument("--e320-source",type=Path,required=True); p.add_argument("--reference-v1-source",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--checkpoint",type=Path,action="append",required=True); p.add_argument("--label-map",type=Path,action="append",required=True); p.add_argument("--training-manifest",type=Path,action="append",required=True); p.add_argument("--registry",type=Path,required=True); a=p.parse_args(argv); print(json.dumps(prepare(queue_path=a.queue,direct_source=a.direct_source,e320_source=a.e320_source,output_root=a.output_root,checkpoint_paths=a.checkpoint,label_maps=a.label_map,training_manifests=a.training_manifest,registry_path=a.registry,reference_v1_source=a.reference_v1_source),sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
