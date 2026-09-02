from pathlib import Path
import json
classes={json.loads(line)["code"]:json.loads(line) for line in Path("outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl").read_text().splitlines()}
spec=[("N05070","P00025","en"),("N05018","P00487","zh")]
rows=[]
for code,part,lang in spec:
 info=classes[code]
 rows.append({"sample_id":f"{code}_{part}_round088_{lang}","target_id":f"rag_round088_{lang}_open_{code}_{part}","source_sample_id":f"{code}_{part}_round088_{lang}","query_image":f"datasets/AgriNet-1K/all/{code}/{code}_{part}.jpg","class_code":code,"class_name":info["english_name"],"class_name_zh":info["chinese_name"],"task_domain":info["task_domain"],"language":lang,"question_type":"open","trajectory_mode":"standard","train_eligible":True,"max_tool_turns":3,"generation_route":"blind_evidence","label_visible_to_teacher":False,"strategy_id":"visual_then_balanced","preferred_sequence":["visual","balanced"],"retrieval_top_k":5,"top_k":5,"preflight_eligible":True,"preflight_target_rank":1,"preflight_target_score":0.87})
source=Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round088_source.jsonl")
plan=Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round088_plan.jsonl")
payload="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows)
source.write_text(payload)
plan.write_text(payload)
print(json.dumps({"source":str(source),"plan":str(plan),"rows":[r["sample_id"] for r in rows]}))
