from pathlib import Path
import glob, json

used = set()
for p in glob.glob("outputs/experiments/rag_sft_iteration/candidates/**/train/*.jsonl", recursive=True):
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        try:
            used.update(json.loads(line).get("images") or [])
        except json.JSONDecodeError:
            pass
classes = {json.loads(line)["code"]: json.loads(line) for line in Path("outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl").read_text(encoding="utf-8").splitlines()}
rows = []
for code, info in classes.items():
    if info.get("task_domain") != "disease":
        continue
    for image in sorted(Path("datasets/AgriNet-1K/all", code).glob(code + "_*.jpg")):
        if str(image) in used:
            continue
        part = image.stem.split("_")[-1]
        rows.append({"sample_id": f"disease_{code}_{part}", "target_id": f"rag_open-round080-disease_{code}_{part}", "source_sample_id": f"disease_{code}_{part}", "query_image": str(image), "class_code": code, "class_name": info.get("english_name"), "class_name_zh": info.get("chinese_name"), "task_domain": "disease", "language": "en", "question_type": "open", "trajectory_mode": "standard", "train_eligible": True, "max_tool_turns": 3, "generation_route": "blind_evidence", "label_visible_to_teacher": False, "strategy_id": "visual_then_balanced", "preferred_sequence": ["visual", "balanced"], "retrieval_top_k": 5})
        if len(rows) >= 96:
            break
    if len(rows) >= 96:
        break
out = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round080_disease_targets.jsonl")
out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
print(json.dumps({"output": str(out), "targets": len(rows)}))
