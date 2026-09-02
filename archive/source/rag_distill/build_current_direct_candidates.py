#!/usr/bin/env python3
"""Rebuild compact Direct candidates against the current evaluation/RAG exclusions."""
from __future__ import annotations
import hashlib,json,re
from collections import Counter
from pathlib import Path
from agrinet.data.sft_recovery import build_direct_pilot,read_jsonl,write_jsonl
from agrinet.research.shared.catalog import ROOT,evaluation_images,normalized
from archive.source.rag_distill.validate_semantic_quality import semantic_row_errors
OUT=ROOT/"outputs/experiments/rag_sft_iteration/strict_candidate_view_v1/direct_current_build"
DATA=ROOT/"outputs/experiments/rag_sft_iteration/strict_candidate_view_v1/direct.current_contract.jsonl"
REPORT=ROOT/"outputs/experiments/rag_sft_iteration/reviews/direct_current_contract_v1_gate.json"
DIRECT=ROOT/"outputs/vlm_sft/disease_pest_large/sft_messages_en_zh.jsonl"
CANDIDATES=ROOT/"outputs/vlm_data/disease_pest_large/contrast_samples_vit_base.jsonl"
RAG=ROOT/"outputs/experiments/rag_sft_iteration/strict_candidate_view_v1/rag.current_contract.jsonl"
def main():
 evals=evaluation_images(); OUT.mkdir(parents=True,exist_ok=True); union=OUT/"evaluation_union.jsonl"
 write_jsonl(union,({"image_path":p} for p in sorted(evals)))
 rag=read_jsonl(RAG); rag_images={str(normalized((r.get("images") or [None])[0])) for r in rag}
 stats=build_direct_pilot(DIRECT,CANDIDATES,union,OUT,ROOT,excluded_rag_images=rag_images)
 rows=read_jsonl(OUT/"accepted/direct_open.jsonl")+read_jsonl(OUT/"accepted/direct_option.jsonl")
 for row in rows:
  metadata=dict(row.get("metadata") or {})
  metadata.update({"data_provenance":"historical_rerendered","source_artifact":str(DIRECT.relative_to(ROOT)),"current_contract_revalidated":True})
  row["metadata"]=metadata
 write_jsonl(DATA,rows); issues=[]; cells=Counter(); letters=Counter(); images=[]
 for row in rows:
  m=row.get("metadata") or {}; assistant="\n".join(str(x.get("content") or "") for x in row.get("messages",[]) if x.get("role")=="assistant"); image=str(normalized((row.get("images") or [None])[0])); images.append(image); cell=(m.get("language"),m.get("task_domain"),m.get("question_type")); cells[cell]+=1; errs=semantic_row_errors(row)
  if not 300<=len(assistant)<=1000: errs.append(f"assistant_length:{len(assistant)}")
  match=re.search(r"<answer>(.*?)</answer>",assistant,re.I|re.S); answer=match.group(1).strip() if match else ""
  if m.get("question_type")=="option":
   letters[answer]+=1
   if answer!=m.get("correct_option") or answer not in "ABCD" or len(answer)!=1: errs.append("option_contract")
  elif answer in {"A","B","C","D"}: errs.append("open_contract")
  if errs: issues.append({"sample_id":row.get("sample_id"),"errors":errs})
 expected_cells={(l,d,q):4 for l in ("en","zh") for d in ("disease","pest") for q in ("open","option")}
 overlaps={"duplicate_images":len(images)-len(set(images)),"rag_images":len(set(images)&rag_images),"evaluation_images":len(set(images)&evals)}
 valid=len(rows)==32 and dict(cells)==expected_cells and letters==Counter({x:4 for x in "ABCD"}) and not issues and not any(overlaps.values())
 report={"schema_version":"agrinet.direct-current-gate/v1","rows":len(rows),"source":str(DIRECT.relative_to(ROOT)),"renderer":"agrinet.data.sft_recovery.build_direct_pilot","evaluation_exclusions":len(evals),"rag_exclusions":len(rag_images),"coverage":{"/".join(k):v for k,v in sorted(cells.items())},"option_letters":dict(sorted(letters.items())),"overlaps":overlaps,"semantic_or_contract_issues":issues,"data_sha256":hashlib.sha256(DATA.read_bytes()).hexdigest(),"valid":valid,"training_authorized":False}
 REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n"); print(json.dumps({"data":str(DATA),"report":str(REPORT),"stats":stats,"valid":valid,"overlaps":overlaps},ensure_ascii=False)); raise SystemExit(0 if valid else 1)
if __name__=="__main__": main()
