from pathlib import Path
import json

classes={json.loads(line)["code"]:json.loads(line) for line in Path("outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl").read_text().splitlines()}
spec=[("N04071","P00010","en","open"),("N04071","P00020","zh","open"),("N04063","P00009","en","option"),("N04063","P00010","zh","option"),("N05070","P00023","en","open"),("N05018","P00484","zh","open")]
codes=["N04071","N04063","N04029","N05070"]
cands=[{"code":c,"name":classes[c]["english_name"],"chinese_name":classes[c]["chinese_name"],"task_domain":classes[c]["task_domain"]} for c in codes]
rows=[]
for code,part,lang,qt in spec:
 info=classes[code]; row={"sample_id":f"{code}_{part}_round087_{lang}","target_id":f"rag_round087_{lang}_{qt}_{code}_{part}","source_sample_id":f"{code}_{part}_round087_{lang}","query_image":f"datasets/AgriNet-1K/all/{code}/{code}_{part}.jpg","class_code":code,"class_name":info["english_name"],"class_name_zh":info["chinese_name"],"task_domain":info["task_domain"],"language":lang,"question_type":qt,"trajectory_mode":"standard","train_eligible":True,"max_tool_turns":3,"generation_route":"blind_evidence","label_visible_to_teacher":False,"strategy_id":"visual_then_balanced","preferred_sequence":["visual","balanced"],"retrieval_top_k":5,"top_k":5,"preflight_eligible":True,"preflight_target_rank":1,"preflight_target_score":0.88}
 if qt=="option": row.update({"candidate_labels":cands,"final_label":code,"final_label_zh":info["chinese_name"],"correct_option":"B"})
 rows.append(row)
source=Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round087_source.jsonl");plan=Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round087_plan.jsonl");payload="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows);source.write_text(payload);plan.write_text(payload)
print(json.dumps({"source":str(source),"plan":str(plan),"rows":[{k:r[k] for k in ("sample_id","language","task_domain","question_type")} for r in rows]},ensure_ascii=False))
