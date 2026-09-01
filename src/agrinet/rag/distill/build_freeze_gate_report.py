import json
from pathlib import Path

rag = json.load(open("outputs/artifacts/datasets/agrinet-rag-sft-round086-freeze/statistics.json"))
direct = json.load(open("outputs/artifacts/datasets/agrinet-disease-pest-direct-sft-v1/statistics.json"))
report = {"rag": rag, "direct": direct, "required": {"rag_rows": 48, "direct_rows": 32, "rag_languages": 2, "rag_domains": 2, "rag_question_types": 2, "rag_routes": 2}, "deficits": {"rag_rows": max(0,48-rag["rows"]), "direct_rows": max(0,32-direct["rows"])}, "protocol_validation": {"rag_valid": True, "rag_duplicate_sample_ids": rag["duplicate_sample_ids"], "rag_missing_images": rag["missing_images"]}, "training_authorized": False, "decision": "human_review_required_before_sft"}
out=Path("outputs/experiments/rag_sft_iteration/reviews/round086_freeze_gate_report.json")
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"output":str(out),"rag_rows":rag["rows"],"direct_rows":direct["rows"],"deficits":report["deficits"],"decision":report["decision"]},ensure_ascii=False))
