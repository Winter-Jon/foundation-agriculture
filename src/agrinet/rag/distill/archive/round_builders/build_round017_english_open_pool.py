#!/usr/bin/env python3
import json
from pathlib import Path
manifest = Path("outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl")
eval_images = {Path(json.loads(line).get("image_path", "")).stem for line in manifest.open(encoding="utf-8") if line.strip()}
used = {"N04053", "N04029", "N04063", "N05070", "N05023", "N05018", "N05028", "N05013", "N05014", "N05010", "N04070"}
rows = []
seen = set()
for line in manifest.open(encoding="utf-8"):
    if not line.strip(): continue
    item = json.loads(line)
    if item.get("language") != "en" or item.get("question_type") != "open": continue
    code = str(item.get("label_code") or ""); image = str(item.get("image_path") or "")
    if not code or code in used or image in seen or not Path(image).is_file(): continue
    fresh = sorted(Path("datasets/AgriNet-1K/all", code).glob(f"{code}_*.jpg"))
    fresh = [p for p in fresh if p.stem not in eval_images]
    if not fresh: continue
    image = str(fresh[0]); seen.add(image); domain = item.get("task_domain"); sid = f"{domain}_{code}_{Path(image).stem.split("_")[-1]}"
    zh = next((a for a in item.get("label_aliases", []) if any("\u4e00" <= c <= "\u9fff" for c in a)), "")
    rows.append({"target_id": f"rag_open-en-{domain}_{Path(image).stem}", "source_sample_id": sid, "query_image": image, "class_code": code, "class_name": item.get("label_name"), "class_name_zh": zh, "task_domain": domain, "language": "en", "question_type": "open", "trajectory_mode": "standard", "train_eligible": True, "max_tool_turns": 3, "reserve": False, "generation_route": "blind_evidence", "label_visible_to_teacher": False, "strategy_id": "visual_then_balanced", "preferred_sequence": ["visual", "balanced"], "retrieval_top_k": 5, "round": 17, "focus": ["english_open_support", "fresh_query_image", "evidence_grounded_open_answer"], "sample_id": sid})
    if len(rows) >= 16: break
out = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round017_english_open_targets.jsonl")
serialized = chr(10).join(json.dumps(r, ensure_ascii=False) for r in rows) + chr(10)
out.write_text(serialized, encoding="utf-8")
print(json.dumps({"rows": len(rows), "domains": sorted({r["task_domain"] for r in rows}), "path": str(out)}, ensure_ascii=False))
