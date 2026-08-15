import json
from pathlib import Path
root=Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan")
classes={json.loads(l)["code"]:json.loads(l) for l in open("outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl")}
classes.update({
  "N04059": {"task_domain": "disease", "english_name": "Grape Downy mildew leaf", "chinese_name": "葡萄霜霉病"},
  "N04053": {"task_domain": "disease", "english_name": "Fig Blight leaf disease", "chinese_name": "无花果叶枯病"},
  "N05028": {"task_domain": "pest", "english_name": "Hebomoia glaucippe", "chinese_name": "蓝幽蝶"},
})
specs=[("N04059","en","P00002"),("N04053","zh","P00002"),("N05070","en","P00005"),("N05028","zh","P00003")]
rows=[]
for code,lang,pid in specs:
 item=classes[code]; image=Path("datasets/AgriNet-1K/all",code)/f"{code}_{pid}.jpg"
 if not image.is_file(): continue
 source_id=f"{item["task_domain"]}_{code}_{pid}"
 rows.append({"target_id":f"rag_open-{lang}-{item["task_domain"]}_{code}_{pid}","source_sample_id":source_id,"query_image":str(image),"class_code":code,"class_name":item["english_name"],"class_name_zh":item["chinese_name"],"task_domain":item["task_domain"],"language":lang,"question_type":"open","trajectory_mode":"standard","train_eligible":True,"max_tool_turns":3,"reserve":False,"generation_route":"blind_evidence","label_visible_to_teacher":False,"strategy_id":"visual_then_balanced","preferred_sequence":["visual","balanced"],"retrieval_top_k":5,"round":15,"focus":["balanced_language_domain_cell","fresh_query_image","evidence_grounded_open_answer"],"sample_id":source_id})
(root/"round015_balanced_targets.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
print(len(rows))
