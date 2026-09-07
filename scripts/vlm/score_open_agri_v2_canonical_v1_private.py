#!/usr/bin/env python3
"""Offline-only scorer for the approved OpenAgri canonical-v1 taxonomy."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from agrinet.research.open_agri_v2_canonical.registry import CANONICAL_VERSION, load_registry


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def answer_normalizer() -> Any:
    path = REPO_ROOT / "vlm/eval/tools/normalize_answers.py"
    spec = importlib.util.spec_from_file_location("canonical_v1_answer_normalizer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    args = parser.parse_args()
    registry = load_registry(args.registry, args.approval, require_approval=True)
    predictions = read_jsonl(args.predictions)
    truth = {str(row["id"]): row for row in read_jsonl(args.truth)}
    ids = [str(row.get("id") or "") for row in predictions]
    if not ids or len(ids) != len(set(ids)) or set(ids) != set(truth):
        raise ValueError("predictions must exactly and uniquely cover canonical-v1 private truth IDs")
    normalizer = answer_normalizer()
    groups: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        item_id = str(prediction["id"])
        private = truth[item_id]
        expected = str(private["canonical_class_code"])
        final, extraction = normalizer.extract_final_answer(str(prediction.get("prediction") or ""))
        predicted = registry.resolve_answer(normalizer.normalize_text(final))
        correct = bool(predicted == expected and not prediction.get("error"))
        group = (str(prediction.get("language") or ""), str(private.get("evaluation_bucket") or ""))
        groups[group].update(rows=1, correct=int(correct), errors=int(bool(prediction.get("error"))))
        rows.append({
            "id": item_id, "language": prediction.get("language"), "route": prediction.get("route"),
            "final_answer_text": final, "answer_extraction_source": extraction,
            "predicted_canonical_class_code": predicted, "correct": correct,
            "protocol_error": bool(prediction.get("error")) or bool((prediction.get("protocol") or {}).get("has_protocol_event")),
        })
    total = Counter()
    by_group = {}
    for group, values in sorted(groups.items()):
        total.update(values)
        by_group["/".join(group)] = {"rows": values["rows"], "accuracy": values["correct"] / values["rows"], "request_error_rate": values["errors"] / values["rows"]}
    metrics = {
        "schema_version": "agrinet.open-agri-v2-canonical-v1-private-score/v1",
        "taxonomy_version": CANONICAL_VERSION, "registry_sha256": registry.digest,
        "private_truth_used_offline_only": True, "private_labels_or_codes_emitted": False,
        "rows": total["rows"], "accuracy": total["correct"] / total["rows"],
        "request_error_rate": total["errors"] / total["rows"], "by_language_and_bucket": by_group,
    }
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
