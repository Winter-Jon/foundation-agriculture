import json
from pathlib import Path
from agrinet.rag.distill.preflight_recovery_targets import search

classes = {json.loads(line)["code"]: json.loads(line) for line in Path("outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl").read_text().splitlines()}
for code in ("N04029", "N04063"):
    info = classes[code]
    image = next(iter(sorted(Path("datasets/AgriNet-1K/all", code).glob(code + "_*.jpg"))))
    print(code, info["english_name"])
    for typ in ("visual", "semantic", "balanced", "rrf"):
        hits = search("http://127.0.0.1:8077", {"query_image": str(image)}, typ, 10)
        print(typ, [(h.get("rank"), h.get("english_name"), round(float(h.get("distance") or 0), 2)) for h in hits[:6]])
