#!/usr/bin/env python3
"""Strictly score fixed-seed public RAG predictions and aggregate class evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vlm/eval/tools"))
from normalize_answers import STRICT_SCORING_POLICY, score_row  # noqa: E402

V2_ROOT = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v2"
DEFAULT_EVIDENCE = REPO_ROOT / "outputs/artifacts/datasets/open-agri-v3-rag-role-evidence-v1"
CATALOG = REPO_ROOT / "outputs/artifacts/datasets/m1-direct-current-hcv-v1/classes.jsonl"
EXPECTED_PROTOCOL = "agrinet.hermes-rag-sglang-async/v5-native-json-multi-query-recovery-strict-similar-classes"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def wilson_lower(correct: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    p = correct / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (centre - margin) / denom


def protocol_error(row: dict[str, Any]) -> bool:
    protocol = row.get("protocol") if isinstance(row.get("protocol"), dict) else {}
    return bool(row.get("error")) or bool(protocol.get("protocol_errors")) or not str(row.get("prediction") or "").strip()


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    evidence = args.evidence.resolve()
    plan = json.loads((evidence / "run_plan.json").read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in plan["seeds"]]
    public = read_jsonl(evidence / "manifests/rag_evidence_public_test.jsonl")
    truth = {str(row["id"]): row for row in read_jsonl(args.v2_root / "vlm_data/accepted/private/test_truth.jsonl")}
    catalog = {str(row["code"]): row for row in read_jsonl(args.catalog)}
    images = {str(row["image_sha256"]): row for row in read_jsonl(args.v2_root / "manifests/images.jsonl")}
    expected_ids = {str(row["id"]) for row in public}
    if not expected_ids or set(truth) != expected_ids:
        raise ValueError("the v2 test public/private ID contract is not a non-empty bijection")

    all_scored: list[dict[str, Any]] = []
    evidence_state: list[dict[str, Any]] = []
    invariant_values: dict[str, set[str]] = defaultdict(set)
    for seed in seeds:
        path = evidence / "predictions" / f"seed-{seed}" / "predictions.jsonl"
        rows = read_jsonl(path) if path.is_file() else []
        by_id: dict[str, dict[str, Any]] = {}
        duplicates: set[str] = set()
        for row in rows:
            item_id = str(row.get("id") or "")
            if item_id in by_id:
                duplicates.add(item_id)
            by_id[item_id] = row
        missing = sorted(expected_ids - set(by_id))
        unexpected = sorted(set(by_id) - expected_ids)
        evidence_state.append({
            "seed": seed, "prediction_path": str(path), "rows": len(rows), "missing_ids": missing,
            "unexpected_ids": unexpected, "duplicate_ids": sorted(duplicates),
            "complete": not missing and not unexpected and not duplicates and len(rows) == len(expected_ids),
        })
        for item_id in expected_ids & set(by_id):
            prediction = by_id[item_id]
            private = truth[item_id]
            code = str(private["class_code"])
            label = catalog[code]
            enriched = {
                **prediction, "id": item_id, "seed": seed, "label_code": code,
                "label_name": label["english_name"], "label_aliases": [label["english_name"], label["chinese_name"]],
                "question_type": "open", "task_domain": private["domain"],
            }
            scored = score_row(enriched, STRICT_SCORING_POLICY)
            scored["strict_protocol_error"] = protocol_error(prediction)
            scored["correct"] = bool(scored["correct"]) and not scored["strict_protocol_error"]
            scored["test_id"] = item_id
            scored["image_sha256"] = str(private["image_sha256"])
            all_scored.append(scored)
            for field in ("rag_protocol_version", "system_prompt_sha256", "tool_schema_sha256", "model_identifier", "temperature", "top_p"):
                value = prediction.get(field)
                invariant_values[field].add(json.dumps(value, ensure_ascii=False, sort_keys=True))
    write_jsonl(evidence / "scored_predictions.jsonl", sorted(all_scored, key=lambda row: (str(row["test_id"]), int(row["seed"]))))

    records_by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_scored:
        records_by_code[str(row["label_code"])].append(row)
    train_counts = Counter(
        row["class_code"] for row in images.values() if row.get("image_split") in {"train_candidate", "dev"}
    )
    classes: list[dict[str, Any]] = []
    incomplete_ids_by_seed = {entry["seed"]: set(entry["missing_ids"]) for entry in evidence_state}
    for code, label in sorted(catalog.items()):
        records = records_by_code[code]
        expected = [item_id for item_id, item in truth.items() if item["class_code"] == code]
        complete = len(records) == len(expected) * len(seeds) and not any(item_id in incomplete_ids_by_seed[seed] for seed in seeds for item_id in expected)
        correct = sum(bool(row["correct"]) for row in records)
        by_seed = {str(seed): [row for row in records if int(row["seed"]) == seed] for seed in seeds}
        seed_accuracy = {seed: (sum(bool(row["correct"]) for row in rows) / len(rows) if rows else None) for seed, rows in by_seed.items()}
        values = [value for value in seed_accuracy.values() if value is not None]
        total = len(records)
        classes.append({
            "class_code": code, "domain": label["task_domain"], "test_images": len(expected), "total_predictions": total,
            "expected_predictions": len(expected) * len(seeds), "correct_predictions": correct,
            "rag_accuracy": correct / total if total else None, "wilson_lower_95": wilson_lower(correct, total) if total else 0.0,
            "seed_accuracy": seed_accuracy,
            "seed_accuracy_variance": sum((value - sum(values) / len(values)) ** 2 for value in values) / len(values) if values else None,
            "protocol_error_rate": sum(bool(row["strict_protocol_error"]) for row in records) / total if total else 1.0,
            "mean_tool_turns": sum(int(row.get("tool_turns") or 0) for row in records) / total if total else 0.0,
            "train_candidate_images": int(train_counts[code]), "evidence_status": "complete" if complete else "evidence-incomplete",
            "evidence_missing_predictions": len(expected) * len(seeds) - total,
        })
    write_jsonl(evidence / "class_difficulty.jsonl", classes)
    summary = {
        "schema_version": "agrinet.open-agri-v3.rag-role-aggregation/v1", "protocol_version": EXPECTED_PROTOCOL,
        "prediction_rows": len(all_scored), "expected_prediction_rows": len(expected_ids) * len(seeds),
        "seed_evidence": evidence_state, "class_count": len(classes),
        "all_classes_have_explicit_state": len(classes) == 217,
        "protocol_invariants": {field: sorted(values) for field, values in invariant_values.items()},
        "protocol_invariants_singleton": all(len(values) == 1 for values in invariant_values.values()),
        "sources": {"v2_private_truth_sha256": sha256(args.v2_root / "vlm_data/accepted/private/test_truth.jsonl"), "catalog_sha256": sha256(args.catalog)},
    }
    (evidence / "aggregation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--v2-root", type=Path, default=V2_ROOT)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    return parser.parse_args()


if __name__ == "__main__":
    result = aggregate(parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
