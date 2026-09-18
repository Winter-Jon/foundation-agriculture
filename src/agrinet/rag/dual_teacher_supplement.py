"""Prepare independent, quota-targeted dual-teacher supplement sources."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from agrinet.rag.e344_pipeline import write, write_rows
from agrinet.rag.e344_prepare import IDENTITIES, digest, rows


def select(config):
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(config["source_root"])
    deficit_path = Path(config.get("deficits", source_root / "live/quota-deficits.json"))
    deficits = [row for row in json.loads(deficit_path.read_text()) if row["deficit"]]
    prior_paths = [Path(path) for path in config.get("prior_sources", [source_root / "pilot-source.jsonl"])]
    prior = [row for path in prior_paths for row in rows(path)]
    used = {key: {row[key] for row in prior} for key in IDENTITIES}
    inventory = list(rows(config["inventory"]))
    multiplier = int(config.get("multiplier", 2))
    if multiplier < 1:
        raise ValueError("invalid_multiplier")
    grouped = defaultdict(list)
    for row in inventory:
        grouped[row["canonical_class_code"]].append(row)
    selected, shortages = [], []
    for deficit in sorted(deficits, key=lambda row: (row["canonical_class_code"], row["arm"], row["question_type"])):
        requested = deficit["deficit"] * multiplier
        coverage = Counter()
        count = 0
        for _ in range(requested):
            candidates = [row for row in grouped[deficit["canonical_class_code"]]
                          if deficit["arm"] in row["bindings"]
                          and all(row[key] not in used[key] for key in IDENTITIES)]
            if not candidates:
                break
            row = min(candidates, key=lambda value: (coverage[value["cluster"]], value["center_distance"], value["image_sha256"]))
            item = {**row, "arm": deficit["arm"], "question_type": deficit["question_type"],
                    "selection_reason": "quota_deficit_oversample_least_covered_cluster"}
            selected.append(item)
            for key in IDENTITIES:
                used[key].add(row[key])
            coverage[row["cluster"]] += 1
            count += 1
        shortages.append({**deficit, "oversample_requested": requested, "selected": count,
                          "remaining_selection_shortage": requested - count})
    for key in IDENTITIES:
        if len({row[key] for row in selected}) != len(selected):
            raise ValueError("supplement_identity_collision:" + key)
    for name in ("classifier-bindings.json", "isolation-audit.json"):
        shutil.copyfile(source_root / name, output / name)
    write_rows(output / "pilot-selection.jsonl", selected)
    write(output / "selection-deficits.json", shortages)
    write(output / "selection-manifest.json", {
        "source_root": str(source_root), "source_selection_sha256": digest(source_root / "pilot-selection.jsonl"),
        "quota_deficits_sha256": digest(deficit_path),
        "inventory_sha256": digest(config["inventory"]), "rows": len(selected),
        "requested": sum(row["oversample_requested"] for row in shortages),
        "selection_shortage": sum(row["remaining_selection_shortage"] for row in shortages),
        "previous_identity_count": {key: len({row[key] for row in prior}) for key in IDENTITIES},
        "prior_source_sha256": {str(path): digest(path) for path in prior_paths},
        "multiplier": multiplier,
    })
    return selected, shortages


if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    chosen, shortages = select(yaml.safe_load(Path(args.config).read_text())["parameters"])
    print(json.dumps({"selected": len(chosen), "selection_shortage": sum(x["remaining_selection_shortage"] for x in shortages)}))
