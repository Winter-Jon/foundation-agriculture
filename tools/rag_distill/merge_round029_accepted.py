import json
from pathlib import Path

sources=[Path("outputs/experiments/rag_sft_iteration/candidates/round-024-evidence-quality-api3-gpt4o/train/agent_sft.accepted.jsonl"),Path("outputs/experiments/rag_sft_iteration/candidates/round-028-stratified-pilot/train/agent_sft.accepted.jsonl"),Path("outputs/experiments/rag_sft_iteration/candidates/round-029-option-pilot/train/agent_sft.accepted.jsonl")]
rows=[]; seen_ids=set(); seen_images=set()
for source in sources:
    for line in source.open(encoding="utf-8"):
        if not line.strip(): continue
        row=json.loads(line); sid=str(row.get("sample_id") or ""); image=str((row.get("images") or [""])[0])
        if not sid or sid in seen_ids: raise SystemExit(f"duplicate sample_id: {sid}")
        if not image or image in seen_images: raise SystemExit(f"duplicate query image: {image}")
        seen_ids.add(sid); seen_images.add(image); rows.append(row)
out=Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round029_accepted17_source.jsonl")
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows),encoding="utf-8")
print(json.dumps({"rows":len(rows),"unique_sample_ids":len(seen_ids),"unique_query_images":len(seen_images)}))
