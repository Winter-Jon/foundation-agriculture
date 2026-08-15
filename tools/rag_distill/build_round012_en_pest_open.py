import json
from pathlib import Path
root=Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan")
code="N05018"
image=sorted(Path("datasets/AgriNet-1K/all",code).glob("*.jpg"))[0]
source_id=f"pest_{image.stem}"
base={"target_id":f"rag_open-en-pest_{image.stem}","source_sample_id":source_id,"query_image":str(image),"class_code":code,"class_name":"cosmopolites sordidus","class_name_zh":"香蕉象甲","task_domain":"pest","language":"en","question_type":"open","trajectory_mode":"standard","train_eligible":True,"max_tool_turns":3,"reserve":False,"generation_route":"blind_evidence","label_visible_to_teacher":False,"strategy_id":"balanced_then_name","preferred_sequence":["balanced","name"],"retrieval_top_k":5,"round":12,"focus":["complementary_english_route","evidence_grounded_open_answer","language_isolation"],"sample_id":source_id}
plan=dict(base); plan["sample_id"]=base["target_id"]+"-round012"; plan["candidate_index"]=1
(root/"round012_en_pest_open_source.jsonl").write_text(json.dumps(base,ensure_ascii=False)+"\n")
(root/"round012_en_pest_open_plan.jsonl").write_text(json.dumps(plan,ensure_ascii=False)+"\n")
print(image)
