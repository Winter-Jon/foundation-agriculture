import json
from pathlib import Path
from tools.rag_distill.preflight_recovery_targets import search

classes = {json.loads(line)["code"]: json.loads(line) for line in Path("outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl").read_text().splitlines()}
report = []
for code in ("N04007", "N04029", "N04063", "N04071"):
    info = classes[code]
    image = next(iter(sorted(Path("datasets/AgriNet-1K/all", code).glob(code + "_*.jpg"))))
    hits = search("http://127.0.0.1:8077", {"query_image": str(image)}, "visual", 10)
    target = next((h for h in hits if h.get("code") == code), None)
    report.append({"code": code, "image": str(image), "target_rank": target.get("rank") if target else None, "target_score": round(float(target.get("distance") or 0), 2) if target else None, "top10": [{"rank": h.get("rank"), "code": h.get("code"), "name": h.get("english_name"), "score": round(float(h.get("distance") or 0), 2)} for h in hits]})
out = Path("outputs/experiments/rag_sft_iteration/reviews/round081_disease_top10_diagnostic.json")
out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False))
