import json
from pathlib import Path
root=Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan")
classes={json.loads(l)["code"]:json.loads(l) for l in open("outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl")}
rows=[]
for code,idxs in {"N05014":[1],"N05018":[2],"N05029":[2],"N05070":[2,3,4]}.items():
  for idx in idxs:
    image=Path("datasets/AgriNet-1K/all",code)/f"{code}_P{idx:05d}.jpg"
    if not image.is_file(): continue
    item=classes[code]; source_id=f"pest_{image.stem}"
    rows.append({"target_id":f"rag_open-en-pest_{image.stem}","source_sample_id":source_id,"query_image":str(image),"class_code":code,"class_name":item["english_name"],"class_name_zh":item["chinese_name"],"task_domain":"pest","language":"en","question_type":"open","trajectory_mode":"standard","train_eligible":True,"max_tool_turns":3,"reserve":False,"generation_route":"blind_evidence","label_visible_to_teacher":False,"strategy_id":"visual_then_balanced","preferred_sequence":["visual","balanced"],"retrieval_top_k":5,"round":13,"focus":["fresh_query_image","evidence_grounded_open_answer","deduplication"],"sample_id":source_id})
(root/"round013_fresh_en_pest_targets.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
print(len(rows))
