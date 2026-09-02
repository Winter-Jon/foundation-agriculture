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
for code in ["N04007","N04029","N04063","N04071","N05014","N05018","N05029","N05070"]:
    item = classes[code]; domain = item["task_domain"]
    images = [p for p in sorted(Path("datasets/AgriNet-1K/all",code).glob(f"{code}_*.jpg")) if str(p) not in used][:4]
    if len(images) < 4: raise SystemExit(f"insufficient fresh images for {code}: {len(images)}")
    for idx, image in enumerate(images):
        lang = "en" if idx % 2 == 0 else "zh"; qtype = "open" if idx in (0,3) else "option"; part=image.stem.split("_")[-1]; sid=f"{domain}_{code}_{part}"
        rows.append({"target_id":f"rag_{qtype}-{lang}-{domain}_{code}_{part}","source_sample_id":sid,"sample_id":sid,"query_image":str(image),"class_code":code,"class_name":item["english_name"],"class_name_zh":item["chinese_name"],"task_domain":domain,"language":lang,"question_type":qtype,"trajectory_mode":"standard","train_eligible":True,"max_tool_turns":3,"reserve":False,"generation_route":"blind_evidence","label_visible_to_teacher":False,"strategy_id":"visual_then_balanced","preferred_sequence":["visual","balanced"],"retrieval_top_k":5,"round":28,"focus":["fresh_unused_image","stratified_pool","strict_evidence_gating"]})
out=Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round028_targets.jsonl")
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
print(json.dumps({"rows":len(rows),"en":sum(r["language"]=="en" for r in rows),"zh":sum(r["language"]=="zh" for r in rows),"disease":sum(r["task_domain"]=="disease" for r in rows),"pest":sum(r["task_domain"]=="pest" for r in rows),"open":sum(r["question_type"]=="open" for r in rows),"option":sum(r["question_type"]=="option" for r in rows)}))
