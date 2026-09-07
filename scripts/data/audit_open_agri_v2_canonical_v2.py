#!/usr/bin/env python3
"""Run deterministic machine and visual audits for open_agri_v3 training data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agrinet.rag.hermes_protocol import normalize_training_messages
from agrinet.rag.tool_schema import TOOL_NAME, validate_tool_arguments
from agrinet.research.open_agri_v2_canonical.registry import load_registry

DATASET = ROOT / "datasets/AgriNet-1K/open_agri_v3"
DEFAULT_OUTPUT = ROOT / "outputs/runs/data/open-agri-v3-audit"


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def write_coverage_csv(path: Path, coverage: Iterable[dict[str, Any]]) -> None:
    entries = list(coverage)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "canonical_class_code", "canonical_english_name", "canonical_chinese_name", "domain",
        "unique_training_images", "training_rows", "direct_rows", "rag_rows",
        "direct_cells", "rag_cells", "image_floor_gap", "direct_route_gap",
        "rag_route_gap", "rag_row_floor_gap",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            row = dict(entry)
            row["direct_cells"] = ";".join(row["direct_cells"])
            row["rag_cells"] = ";".join(row["rag_cells"])
            writer.writerow(row)


def final_answer(row: dict[str, Any]) -> str:
    content = str(row["messages"][-1]["content"])
    return content.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()


def option_codes(row: dict[str, Any], registry: Any) -> dict[str, str | None]:
    user = next(item["content"] for item in row["messages"] if item["role"] == "user")
    result: dict[str, str | None] = {}
    for line in str(user).splitlines():
        line = line.strip()
        if len(line) > 3 and line[0] in "ABCD" and line[1:3] == ". ":
            result[line[0]] = registry.resolve_answer(line[3:])
    return result


def load_known_classes(path: Path) -> dict[str, dict[str, Any]]:
    """Load the benchmark's Known classes from the versioned split manifest."""
    return {
        str(row["canonical_class_code"]): row
        for row in rows(path)
        if row.get("class_role") == "known"
    }


def class_coverage(
    all_rows: Iterable[dict[str, Any]],
    known_classes: dict[str, dict[str, Any]],
    *,
    image_floor: int,
    route_row_floor: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize image, route, and route-cell supervision by Known class."""
    per_code: dict[str, dict[str, Any]] = {
        code: {
            "image_sha256": set(), "direct_rows": 0, "rag_rows": 0,
            "direct_cells": set(), "rag_cells": set(),
        }
        for code in known_classes
    }
    for row in all_rows:
        metadata = row["metadata"]
        code = str(metadata["canonical_class_code"])
        if code not in per_code:
            continue
        counts = per_code[code]
        image_sha256 = metadata.get("v2_image_sha256") or metadata.get("image_sha256")
        if isinstance(image_sha256, str) and image_sha256:
            counts["image_sha256"].add(image_sha256)
        route = str(metadata["route"])
        cell = f"{metadata['question_type']}/{metadata['language']}"
        counts[f"{route}_rows"] += 1
        counts[f"{route}_cells"].add(cell)

    coverage = []
    for code, split in sorted(known_classes.items()):
        counts = per_code[code]
        image_count = len(counts["image_sha256"])
        direct_rows = counts["direct_rows"]
        rag_rows = counts["rag_rows"]
        coverage.append({
            "canonical_class_code": code,
            "canonical_english_name": split["canonical_english_name"],
            "canonical_chinese_name": split["canonical_chinese_name"],
            "domain": split["domain"],
            "unique_training_images": image_count,
            "training_rows": direct_rows + rag_rows,
            "direct_rows": direct_rows,
            "rag_rows": rag_rows,
            "direct_cells": sorted(counts["direct_cells"]),
            "rag_cells": sorted(counts["rag_cells"]),
            "image_floor_gap": image_count < image_floor,
            "direct_route_gap": direct_rows == 0,
            "rag_route_gap": rag_rows == 0,
            "rag_row_floor_gap": rag_rows < route_row_floor,
        })
    image_distribution = Counter(item["unique_training_images"] for item in coverage)
    summary = {
        "known_classes": len(coverage),
        "image_floor": image_floor,
        "route_row_floor": route_row_floor,
        "image_count_distribution": {str(key): image_distribution[key] for key in sorted(image_distribution)},
        "image_floor_gaps": [item["canonical_class_code"] for item in coverage if item["image_floor_gap"]],
        "direct_route_gaps": [item["canonical_class_code"] for item in coverage if item["direct_route_gap"]],
        "rag_route_gaps": [item["canonical_class_code"] for item in coverage if item["rag_route_gap"]],
        "rag_row_floor_gaps": [item["canonical_class_code"] for item in coverage if item["rag_row_floor_gap"]],
    }
    return coverage, summary


def audit_row(row: dict[str, Any], registry: Any) -> list[str]:
    metadata = row.get("metadata") or {}
    errors: list[str] = []
    code = metadata.get("canonical_class_code")
    if not isinstance(code, str) or code not in registry.rows_by_code:
        errors.append("unknown_canonical_class")
        return errors
    answer = final_answer(row)
    if metadata.get("question_type") == "open":
        if registry.resolve_answer(answer) != code:
            errors.append("open_answer_target_mismatch")
    else:
        options = option_codes(row, registry)
        if set(options) != set("ABCD") or any(value is None for value in options.values()):
            errors.append("option_labels_unresolvable")
        elif options.get(answer) != code:
            errors.append("option_answer_target_mismatch")
    if metadata.get("route") == "rag":
        try:
            normalize_training_messages(row["messages"], tool_name=TOOL_NAME,
                                        validate_arguments=validate_tool_arguments,
                                        require_tool_calls=True)
        except ValueError as exc:
            errors.append(f"rag_contract:{exc}")
    return errors


def build_montage(samples: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return {"status": "skipped", "reason": "Pillow unavailable"}
    tiles = []
    for sample in samples:
        image = Path(sample["image_path"])
        if not image.is_file():
            continue
        with Image.open(image) as source:
            tile = source.convert("RGB")
            tile.thumbnail((256, 192))
            canvas = Image.new("RGB", (256, 240), "white")
            canvas.paste(tile, ((256 - tile.width) // 2, 0))
            ImageDraw.Draw(canvas).text((6, 196), sample["caption"][:70], fill="black")
            tiles.append(canvas)
    if not tiles:
        return {"status": "skipped", "reason": "no readable sampled images"}
    columns = 4
    rows_count = (len(tiles) + columns - 1) // columns
    montage = Image.new("RGB", (256 * columns, 240 * rows_count), "white")
    for index, tile in enumerate(tiles):
        montage.paste(tile, ((index % columns) * 256, (index // columns) * 240))
    montage.save(output, quality=90)
    return {"status": "created", "path": str(output), "tiles": len(tiles)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="20260904T033000-open-agri-v3-audit")
    parser.add_argument("--seed", default="open-agri-v3-audit-v1")
    parser.add_argument("--per-cell", type=int, default=2)
    parser.add_argument("--image-floor", type=int, default=4)
    parser.add_argument("--route-row-floor", type=int, default=4)
    args = parser.parse_args()
    dataset = args.dataset_root.resolve()
    output = args.output_root / args.run_id
    if output.exists():
        raise SystemExit(f"audit output exists: {output}")
    registry = load_registry(dataset / "taxonomy/canonical_label_registry.jsonl", dataset / "taxonomy/approval.json")
    all_rows = []
    for path in sorted((dataset / "vlm_data/accepted/train").glob("*.jsonl")):
        all_rows.extend(rows(path))
    known_classes = load_known_classes(dataset / "manifests/class_split.jsonl")
    coverage, coverage_summary = class_coverage(
        all_rows, known_classes, image_floor=args.image_floor, route_row_floor=args.route_row_floor,
    )
    errors = []
    cells: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        metadata = row["metadata"]
        row_errors = audit_row(row, registry)
        if row_errors:
            errors.append({"sample_id": row["sample_id"], "errors": row_errors})
        cells[(metadata["route"], metadata["task_domain"], metadata["question_type"], metadata["language"])].append(row)
    sampled = []
    for cell, values in sorted(cells.items()):
        ordered = sorted(values, key=lambda row: hashlib.sha256(f"{args.seed}:{row['sample_id']}".encode()).hexdigest())
        for row in ordered[:args.per_cell]:
            metadata = row["metadata"]
            sampled.append({
                "sample_id": row["sample_id"], "cell": "/".join(cell), "image_sha256": metadata["v2_image_sha256"],
                "canonical_class_code": metadata["canonical_class_code"],
                "canonical_english_name": metadata["canonical_english_name"],
                "canonical_chinese_name": metadata["canonical_chinese_name"],
                "answer": final_answer(row), "image_path": row["images"][0],
                "caption": f"{metadata['canonical_class_code']} {metadata['canonical_english_name']}",
            })
    write_jsonl(output / "artifacts/machine_errors.jsonl", errors)
    write_jsonl(output / "artifacts/class_coverage.jsonl", coverage)
    write_coverage_csv(output / "artifacts/class_coverage.csv", coverage)
    write_json(output / "artifacts/class_coverage_summary.json", coverage_summary)
    write_jsonl(output / "artifacts/stratified_samples.jsonl", sampled)
    montage = build_montage(sampled, output / "artifacts/stratified_montage.jpg")
    report = {
        "schema_version": "agrinet.open-agri-v3.audit/v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset), "registry_sha256": registry.digest, "seed": args.seed,
        "rows": len(all_rows), "machine_errors": len(errors), "cells": {"/".join(key): len(value) for key, value in sorted(cells.items())},
        "class_coverage": coverage_summary,
        "sampled_rows": len(sampled), "montage": montage,
    }
    write_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
