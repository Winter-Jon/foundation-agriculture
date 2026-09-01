#!/usr/bin/env python3
"""Build a hash-isolated bilingual historical Direct candidate pool."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agrinet.data.rebuild_sft import CELLS, STAGE_SPECS, image_digest, validate_direct_rows
from agrinet.data.sft_recovery import ANSWER_RE, _normalize_direct_think, _option_row, read_jsonl, write_jsonl
from agrinet.rag.distill.catalog_and_isolation import ROOT, exposed_images

DIRECT = ROOT / "outputs/vlm_sft/disease_pest_large/sft_messages_en_zh.jsonl"
CANDIDATES = ROOT / "outputs/vlm_data/disease_pest_large/contrast_samples_vit_base.jsonl"
DIAGNOSTIC = ROOT / "outputs/artifacts/datasets/agrinet-rag-sft-stagea-validation-v3/matched_diagnostic_192.jsonl"
SEED = 20260816


def language(row: dict[str, Any]) -> str:
    user = next((str(message.get("content") or "") for message in row.get("messages") or [] if message.get("role") == "user"), "")
    return "zh" if re.search(r"[\u4e00-\u9fff]", user) else "en"


def domain(sample: dict[str, Any]) -> str:
    return str(sample.get("task_domain") or ("disease" if str(sample.get("final_label") or "").startswith("N04") else "pest"))


def rendered_open(source: dict[str, Any], sample: dict[str, Any], index: int) -> dict[str, Any]:
    lang = language(source)
    messages = [dict(message) for message in source["messages"] if message.get("role") != "system"]
    original = str(messages[-1].get("content") or "")
    answer = ANSWER_RE.search(original)
    labels = sample.get("candidate_labels") or [{}]
    label = str(answer.group(1).strip() if answer else sample.get("final_label_zh") if lang == "zh" else labels[0].get("name") or "")
    messages[-1] = {**messages[-1], "content": _normalize_direct_think(original, lang) + f"\n\n<answer>{label}</answer>"}
    return {"sample_id": f"rebuild-direct-open-{lang}-{sample['sample_id']}", "images": list(source["images"][:1]), "messages": messages, "metadata": {"source_dataset": "historical_checkpoint165_direct", "source_artifact": str(DIRECT.relative_to(ROOT)), "source_row_index": index, "source_sample_id": sample["sample_id"], "language": lang, "task_domain": domain(sample), "question_type": "open", "train_eligible": False, "data_provenance": "historical_rerendered", "current_contract_revalidated": True}}


def isolation_hashes(root: Path) -> set[str]:
    hashes = {image_digest(str(row.get("image_path") or ""), root) for row in read_jsonl(DIAGNOSTIC)}
    for image in exposed_images():
        try:
            hashes.add(image_digest(image, root))
        except Exception:
            continue
    return hashes


def build(stage: str, destination: Path, root: Path = ROOT) -> dict[str, Any]:
    spec = STAGE_SPECS[stage]
    candidates = {str(row["query_image"]): row for row in read_jsonl(CANDIDATES)}
    by_image: dict[str, dict[str, tuple[int, dict[str, Any]]]] = defaultdict(dict)
    excluded = Counter()
    forbidden = isolation_hashes(root)
    source_rows = read_jsonl(DIRECT)
    for index, row in enumerate(source_rows):
        image = str((row.get("images") or [""])[0])
        sample = candidates.get(image)
        if not sample:
            excluded["no_candidate_metadata"] += 1
            continue
        try:
            digest = image_digest(image, root)
        except Exception:
            excluded["missing_image"] += 1
            continue
        if digest in forbidden:
            excluded["image_isolation"] += 1
            continue
        by_image[image][language(row)] = (index, row)
    pairs = [(image, langs["en"], langs["zh"], candidates[image]) for image, langs in by_image.items() if {"en", "zh"} <= set(langs)]
    selected: list[dict[str, Any]] = []
    shortages: dict[str, int] = {}
    used: set[str] = set()
    for question_type in ("open", "option"):
        for task_domain in ("disease", "pest"):
            pool = [pair for pair in pairs if domain(pair[3]) == task_domain and pair[0] not in used]
            random.Random(f"{SEED}:{stage}:{question_type}:{task_domain}").shuffle(pool)
            chosen = pool[:spec["direct_per_cell"]]
            shortages[f"{question_type}/{task_domain}"] = max(0, spec["direct_per_cell"] - len(chosen))
            for pair_index, (image, en, zh, sample) in enumerate(chosen):
                used.add(image)
                for index, source in (en, zh):
                    row = rendered_open(source, sample, index) if question_type == "open" else _option_row(source, sample, "ABCD"[pair_index % 4], index)
                    metadata = dict(row.get("metadata") or {})
                    metadata.update({"image_sha256": image_digest(image, root), "paired_image_id": image, "data_provenance": "historical_rerendered", "current_contract_revalidated": True})
                    row["metadata"] = metadata
                    selected.append(row)
    report = {"stage": stage, "seed": SEED, "source_rows": len(source_rows), "paired_isolated_images": len(pairs), "selected_rows": len(selected), "selected_unique_images": len(used), "shortages": shortages, "exclusions": dict(excluded), "forbidden_image_hashes": len(forbidden), "validation": validate_direct_rows(selected, stage=stage, image_hashes=forbidden), "training_authorized": False}
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "historical_rerendered_direct.jsonl", selected)
    (destination / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGE_SPECS), required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.stage, args.destination)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not any(report["shortages"].values()) and report["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
