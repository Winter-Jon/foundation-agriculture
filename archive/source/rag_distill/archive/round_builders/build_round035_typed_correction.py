"""Select four unused high-margin targets for a bounded correction pilot."""
from __future__ import annotations
import json
from pathlib import Path

root = Path("outputs/experiments/rag_sft_iteration")
plans = sorted((root / "rounds/round_0001/plan").glob("*.jsonl"))
used = set()
for p in (root / "candidates").glob("*/train/agent_sft.accepted.jsonl"):
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            used.add(str(row.get("metadata", {}).get("source_sample_id") or row.get("metadata", {}).get("sample_id") or row.get("sample_id") or ""))
desired = [("en", "disease", "blind_evidence"), ("en", "pest", "oracle_grounded"), ("zh", "disease", "blind_evidence"), ("zh", "pest", "oracle_grounded")]
pool = {}
for p in plans:
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        sid = str(row.get("source_sample_id") or row.get("sample_id") or "")
        if sid in used or not row.get("query_image") or not Path(row["query_image"]).is_file(): continue
        pre = row.get("strategy_preflight", {}).get("visual_then_balanced", {})
        if not (pre.get("eligible") and int(pre.get("rank", 99)) == 1 and float(pre.get("score", 0)) >= 0.70): continue
        key = (str(row.get("language")), str(row.get("task_domain")))
        pool.setdefault(key, []).append(row)
chosen = []
for language, domain, route in desired:
    options = sorted(pool.get((language, domain), []), key=lambda r: (-float(r.get("strategy_preflight", {}).get("visual_then_balanced", {}).get("score", 0)), str(r.get("source_sample_id"))))
    if not options: raise SystemExit(f"no unused rank1 target for {language}/{domain}")
    row = dict(options[0])
    row.update({"sample_id": row.get("source_sample_id"), "target_id": f"rag_correction-enforced-{language}-{domain}-{row.get('source_sample_id')}", "trajectory_mode": "stop_correction", "generation_route": route, "label_visible_to_teacher": route == "oracle_grounded", "strategy_id": "visual_then_balanced", "preferred_sequence": ["visual", "balanced"], "round": 35, "focus": ["typed_correction", "evidence_confirmed", "rank1_margin", "language_isolation"]})
    chosen.append(row)
out = root / "rounds/round_0001/plan/round035_typed_correction_plan.jsonl"
out.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in chosen), encoding="utf-8")
print(json.dumps({"rows": len(chosen), "ids": [r.get("source_sample_id") for r in chosen], "routes": [r["generation_route"] for r in chosen]}, ensure_ascii=False))
