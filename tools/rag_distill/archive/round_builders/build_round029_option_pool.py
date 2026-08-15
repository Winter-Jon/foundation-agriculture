import json
from pathlib import Path

used = set()
for path in Path("outputs/artifacts/datasets").glob("agrinet-rag-sft-round*-singleimg/data.jsonl"):
    for line in path.open():
        try: used.add((json.loads(line).get("images") or [""])[0])
        except Exception: pass
rows=[]; seen=set()
for path in Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan").glob("*.jsonl"):
    for line in path.open():
        try: row=json.loads(line)
        except Exception: continue
        image=row.get("query_image")
        if row.get("question_type") != "option" or not row.get("correct_option") or not image or image in used or image in seen: continue
        seen.add(image); row={**row,"round":29,"generation_route":"blind_evidence","label_visible_to_teacher":False,"focus":["fresh_option","correct_option_coverage","strict_evidence_gating"]}; rows.append(row)
rows.sort(key=lambda r:(r.get("language"),r.get("task_domain"),r.get("correct_option"),r.get("query_image")))
out=Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round029_option_targets.jsonl"); out.parent.mkdir(parents=True,exist_ok=True); out.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
print(json.dumps({"rows":len(rows),"targets":[r.get("target_id") for r in rows],"options":[r.get("correct_option") for r in rows]}))
