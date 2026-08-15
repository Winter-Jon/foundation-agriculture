#!/usr/bin/env python3
import json
from pathlib import Path

root = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan")
classes = {}
for line in Path("outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl").open(encoding="utf-8"):
    if line.strip():
        item = json.loads(line)
        classes[item["code"]] = item
classes.update({"N05070": {"task_domain": "pest", "english_name": "Tipula paludosa", "chinese_name": "湿地蚊"}})
specs = [("N04007", "en", ["P00001", "P00002"]), ("N04029", "zh", ["P00001", "P00002"]), ("N04063", "en", ["P00001", "P00002"]), ("N05070", "zh", ["P00006", "P00007"])]
rows = []
for code, lang, indices in specs:
    item = classes[code]
    for pid in indices:
        image = Path("datasets/AgriNet-1K/all", code, f"{code}_{pid}.jpg")
        if not image.is_file():
            continue
        domain = item["task_domain"]
        sid = f"{domain}_{code}_{pid}"
        rows.append({"target_id": f"rag_open-{lang}-{domain}_{code}_{pid}", "source_sample_id": sid, "query_image": str(image), "class_code": code, "class_name": item["english_name"], "class_name_zh": item["chinese_name"], "task_domain": domain, "language": lang, "question_type": "open", "trajectory_mode": "standard", "train_eligible": True, "max_tool_turns": 3, "reserve": False, "generation_route": "blind_evidence", "label_visible_to_teacher": False, "strategy_id": "visual_then_balanced", "preferred_sequence": ["visual", "balanced"], "retrieval_top_k": 5, "round": 16, "focus": ["language_domain_balance", "fresh_query_image", "evidence_grounded_open_answer"], "sample_id": sid})
out = root / "round016_balanced_targets.jsonl"
out.write_text(chr(10).join(json.dumps(row, ensure_ascii=False) for row in rows) + chr(10), encoding="utf-8")
print(json.dumps({"rows": len(rows), "path": str(out)}, ensure_ascii=False))
