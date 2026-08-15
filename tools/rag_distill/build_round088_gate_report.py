import json
from pathlib import Path
rag=json.load(open("outputs/artifacts/datasets/agrinet-rag-sft-round088-freeze/statistics.json"))
direct=json.load(open("outputs/artifacts/datasets/agrinet-disease-pest-direct-sft-v1/statistics.json"))
report={"rag":rag,"direct_rows":direct["rows"],"required":{"rag_rows":48,"direct_rows":32},"deficits":{"rag_rows":max(0,48-rag["rows"])},"training_authorized":False,"formal_eval_authorized":False,"decision":"freeze_valid_but_human_review_required_before_sft"}
out=Path("outputs/experiments/rag_sft_iteration/reviews/round088_freeze_gate_report.json")
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"output":str(out),"rag_rows":rag["rows"],"direct_rows":direct["rows"],"decision":report["decision"]},ensure_ascii=False))
