#!/usr/bin/env python3
"""Build two fresh Chinese Option-B rows across Blind and Oracle routes."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from archive.source.rag_distill.archive.round_builders.build_round098_option_coverage import ROOT, audited_catalog, candidate_labels
from archive.source.rag_distill.archive.round_builders.build_round111_option_coverage import exposed_images, write_jsonl
ROUND=114; LETTER="B"; TEACHER_TEMPERATURE=0.0
OUT=ROOT/"outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round114_b_chinese_route_smoke"
ARTIFACT=ROOT/"outputs/experiments/rag_sft_iteration/candidates/round114_b_chinese_route_smoke"
SPECS=(("N04106","disease","blind_evidence"),("N05010","pest","oracle_grounded"))
def main():
 classes=audited_catalog(); forbidden=exposed_images(); excluded=len(forbidden); source=[]; plan=[]
 for i,(code,domain,route) in enumerate(SPECS,1):
  folder=ROOT/"datasets/AgriNet-1K/all"/code
  image=next((str(p.relative_to(ROOT)) for p in sorted(folder.glob(f"{code}_P*.jpg")) if str(p.relative_to(ROOT)) not in forbidden),None)
  if not image: raise RuntimeError(f"no fresh Round114 image for {code}")
  info=classes[code]; labels=candidate_labels(classes,code,domain,LETTER)
  if labels[1]["code"]!=code: raise RuntimeError(f"invalid B mapping for {code}")
  sid=f"round114_zh_{domain}_{route}_{code}_{Path(image).stem}"
  row={"sample_id":sid,"source_sample_id":sid,"target_id":f"rag_option-round114-{sid}","query_image":image,"image_sha256":hashlib.sha256((ROOT/image).read_bytes()).hexdigest(),"class_code":code,"class_name":info["english_name"],"class_name_zh":info["chinese_name"],"task_domain":domain,"language":"zh","question_type":"option","candidate_labels":labels,"final_label":code,"final_label_zh":info["chinese_name"],"correct_option":LETTER,"trajectory_mode":"standard","train_eligible":True,"max_tool_turns":3,"generation_route":route,"label_visible_to_teacher":route=="oracle_grounded","strategy_id":"visual_then_balanced","preferred_sequence":["visual","balanced"],"retrieval_top_k":5,"top_k":5,"round":ROUND,"focus":["option_B","chinese","route_isolation","fresh_image"],"teacher_temperature":TEACHER_TEMPERATURE,"preflight_required":True,"preflight_eligible":None}
  source.append(row); plan.append(dict(row,sample_id=f"round114-option-{i}-{sid}")); forbidden.add(image)
 report={"round":ROUND,"status":"static_audit_passed_preflight_required","rows":2,"letter":LETTER,"language":"zh","domains":{"disease":1,"pest":1},"routes":{"blind_evidence":1,"oracle_grounded":1},"unique_query_images":2,"historical_exposed_and_evaluation_images_excluded":excluded,"teacher_temperature":TEACHER_TEMPERATURE,"training_authorized":False,"formal_eval_authorized":False}
 for d in (OUT,ARTIFACT): d.mkdir(parents=True,exist_ok=True); write_jsonl(d/"source.jsonl",source); write_jsonl(d/"plan.jsonl",plan); (d/"static_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
 print(json.dumps({"plan":str(OUT/"plan.jsonl"),"audit":report},ensure_ascii=False))
if __name__=="__main__": main()
