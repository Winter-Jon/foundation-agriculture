import json
from pathlib import Path

classes = {json.loads(x)["code"]: json.loads(x) for x in Path("outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl").open()}
used = set()
for path in Path("outputs/artifacts/datasets").glob("agrinet-rag-sft-round*-singleimg/data.jsonl"):
    for line in path.open():
        try: used.add(str((json.loads(line).get("images") or [""])[0]))
        except Exception: pass
for path in Path("outputs/experiments/rag_sft_iteration/candidates").glob("**/train/*.jsonl"):
    for line in path.open():
        try: used.add(str((json.loads(line).get("images") or [""])[0]))
        except Exception: pass
rows = []
for code in ["N04007","N05014","N04029","N05018","N04063","N05029","N04071","N05070"]:
    item = classes[code]; images = [p for p in sorted(Path("datasets/AgriNet-1K/all",code).glob(f"{code}_*.jpg")) if str(p) not in used]
    for idx, image in enumerate(images[:2]):
        lang = "en" if idx == 0 else "zh"; part = image.stem.split("_")[-1]; domain = item["task_domain"]; qtype = "option" if len(rows)%4 in (1,3) else "open"
        rows.append({"target_id":f"rag_{qtype}-{lang}-{domain}_{code}_{part}","source_sample_id":f"{domain}_{code}_{part}","sample_id":f"{domain}_{code}_{part}","query_image":str(image),"class_code":code,"class_name":item["english_name"],"class_name_zh":item["chinese_name"],"task_domain":domain,"language":lang,"question_type":qtype,"trajectory_mode":"standard","train_eligible":True,"max_tool_turns":3,"reserve":False,"generation_route":"blind_evidence","label_visible_to_teacher":False,"strategy_id":"visual_then_balanced","preferred_sequence":["visual","balanced"],"retrieval_top_k":5,"round":27,"focus":["fresh_unused_image","replay_safe_mixture_diagnostic","strategy_diversity"]})
rows = rows[:8]
out = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round027_targets.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
print(json.dumps({"rows":len(rows),"en":sum(r["language"]=="en" for r in rows),"zh":sum(r["language"]=="zh" for r in rows)}))
