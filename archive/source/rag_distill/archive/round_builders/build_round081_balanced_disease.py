import glob, json
from pathlib import Path
from archive.source.rag_distill.preflight_recovery_targets import search

used = set()
for p in glob.glob("outputs/experiments/rag_sft_iteration/candidates/**/train/*.jsonl", recursive=True):
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        try: used.update(json.loads(line).get("images") or [])
        except json.JSONDecodeError: pass
code = "N04063"
info = {json.loads(line)["code"]: json.loads(line) for line in Path("outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl").read_text().splitlines()}[code]
selected = []
for image in sorted(Path("datasets/AgriNet-1K/all", code).glob(code + "_*.jpg")):
    if str(image) in used: continue
    hits = search("http://127.0.0.1:8077", {"query_image": str(image)}, "balanced", 5)
    names = {str(info.get("english_name", "")).lower(), str(info.get("chinese_name", "")).lower()}
    match = next((h for h in hits if str(h.get("english_name", "")).lower() in names or str(h.get("chinese_name", "")).lower() in names), None)
    if match and int(match.get("rank") or 99) == 1 and float(match.get("distance") or 0) >= .70:
        part = image.stem.split("_")[-1]
        selected.append({"sample_id": f"disease_{code}_{part}", "target_id": f"rag_open-round081-balanced_{code}_{part}", "source_sample_id": f"disease_{code}_{part}", "query_image": str(image), "class_code": code, "class_name": info.get("english_name"), "class_name_zh": info.get("chinese_name"), "task_domain": "disease", "language": "en", "question_type": "open", "trajectory_mode": "standard", "train_eligible": True, "max_tool_turns": 3, "generation_route": "blind_evidence", "label_visible_to_teacher": False, "strategy_id": "balanced_stop", "preferred_sequence": ["balanced"], "retrieval_top_k": 5, "preflight_eligible": True, "preflight_target_rank": 1, "preflight_target_score": round(float(match.get("distance") or 0), 2)})
    if len(selected) >= 2: break
out = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round081_balanced_disease_plan.jsonl")
out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selected), encoding="utf-8")
print(json.dumps({"selected": len(selected), "rows": selected}, ensure_ascii=False))
