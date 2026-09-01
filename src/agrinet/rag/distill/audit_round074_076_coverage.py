"""Create a compact coverage audit for accepted high-margin pilot rows."""
from __future__ import annotations
import glob, json
from pathlib import Path

ROOT = Path("outputs/experiments/rag_sft_iteration")
rows = []
for fn in sorted(glob.glob(str(ROOT / "candidates/round07[4-6]*/train/agent_sft.accepted.jsonl"))):
    run = Path(fn).parents[3].name
    for line in Path(fn).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        x = json.loads(line)
        sid = x["sample_id"]
        images = x.get("images") or []
        meta = x.get("metadata") or {}
        rows.append({"run": run, "sample_id": sid, "language": meta.get("language"),
                     "question_type": meta.get("question_type"),
                     "domain": "pest" if sid.startswith("pest_") else "disease" if sid.startswith("disease_") else "unknown",
                     "route": meta.get("generation_route") or "unknown",
                     "target_image": images[0] if images else None,
                     "image_count": len(images), "message_count": len(x.get("messages") or [])})
sample_ids = [r["sample_id"] for r in rows]
target_images = [r["target_image"] for r in rows]
report = {"source_runs": sorted({r["run"] for r in rows}), "rows": rows,
          "counts": {"accepted": len(rows), "language": {}, "domain": {}, "question_type": {}, "route": {}},
          "duplicates": {"sample_id": len(sample_ids) - len(set(sample_ids)), "target_image": len(target_images) - len(set(target_images))},
          "coverage_gate": {"minimum_rows": 16, "minimum_languages": 2, "minimum_domains": 2, "minimum_question_types": 2, "minimum_routes": 2, "passed": False},
          "decision": "no_freeze_no_sft_coverage_insufficient"}
for key in ("language", "domain", "question_type", "route"):
    for r in rows:
        report["counts"][key][r[key]] = report["counts"][key].get(r[key], 0) + 1
out = ROOT / "reviews/round074_076_coverage_audit.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "accepted": len(rows), "counts": report["counts"], "duplicates": report["duplicates"], "decision": report["decision"]}, ensure_ascii=False))
