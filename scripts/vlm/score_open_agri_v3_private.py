#!/usr/bin/env python3
"""Score OpenAgri v3 predictions against private truth without exporting labels."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from agrinet.research.open_agri_v2_canonical.registry import load_registry


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalizer() -> Any:
    path = ROOT / "vlm/eval/tools/normalize_answers.py"
    spec = importlib.util.spec_from_file_location("open_agri_v3_normalizer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--allow-subset", action="store_true", help="Allow a unique prediction subset for a deterministic smoke manifest.")
    args = parser.parse_args()
    prediction_rows, truth_rows = rows(args.predictions), rows(args.truth)
    truth = {str(row["id"]): row for row in truth_rows}
    ids = [str(row.get("id") or "") for row in prediction_rows]
    if not ids or len(ids) != len(set(ids)) or (not set(ids).issubset(set(truth))) or (not args.allow_subset and set(ids) != set(truth)):
        raise ValueError("predictions must uniquely cover either all v3 private truth IDs or a declared smoke subset")
    registry = load_registry(ROOT / "datasets/AgriNet-1K/open_agri_v3/taxonomy/canonical_label_registry.jsonl",
                             ROOT / "datasets/AgriNet-1K/open_agri_v3/taxonomy/approval.json", require_approval=True)
    extract = normalizer()
    groups: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    redacted = []
    for prediction in prediction_rows:
        private = truth[str(prediction["id"])]
        expected = str(private["canonical_class_code"])
        rendered = str(prediction.get("prediction") or "")
        final, source = extract.extract_final_answer(rendered)
        # A trajectory that finishes by asking for another tool invocation has
        # not produced an answer.  The standard evaluation policy counts that
        # sample as incorrect (rather than invalidating the whole run).
        terminal_tool_call = "<tool_call>" in rendered
        protocol_error = bool(prediction.get("error")) or terminal_tool_call
        correct = registry.resolve_answer(extract.normalize_text(final)) == expected and not protocol_error
        key = (str(prediction.get("language") or ""), str(private.get("domain") or ""), str(private.get("evaluation_bucket") or ""))
        groups[key].update(rows=1, correct=int(correct), errors=int(bool(prediction.get("error"))), terminal_tool_calls=int(terminal_tool_call))
        buckets[str(private.get("evaluation_bucket") or "")].update(
            rows=1, correct=int(correct), errors=int(bool(prediction.get("error"))), terminal_tool_calls=int(terminal_tool_call)
        )
        redacted.append({"id": prediction["id"], "language": prediction.get("language"), "route": prediction.get("route"),
                         "final_answer_text": final, "answer_extraction_source": source, "correct": correct,
                         "protocol_error": protocol_error, "terminal_tool_call": terminal_tool_call})
    total = Counter(); by_group = {}
    for key, values in sorted(groups.items()):
        total.update(values)
        by_group["/".join(key)] = {"rows": values["rows"], "accuracy": values["correct"] / values["rows"], "request_error_rate": values["errors"] / values["rows"], "terminal_tool_call_rate": values["terminal_tool_calls"] / values["rows"]}
    by_bucket = {
        bucket: {
            "rows": values["rows"],
            "accuracy": values["correct"] / values["rows"],
            "request_error_rate": values["errors"] / values["rows"],
            "terminal_tool_call_count": values["terminal_tool_calls"],
        }
        for bucket, values in sorted(buckets.items())
    }
    metrics = {"schema_version": "agrinet.open-agri-v3-private-score/v2", "rows": total["rows"],
               "accuracy": total["correct"] / total["rows"], "request_error_rate": total["errors"] / total["rows"],
               "terminal_tool_call_count": total["terminal_tool_calls"],
               "terminal_tool_call_policy": "count_as_incorrect",
               "registry_sha256": registry.digest, "private_truth_used_offline_only": True,
               "private_labels_or_codes_emitted": False, "by_language_domain_bucket": by_group}
    metrics["by_evaluation_bucket"] = by_bucket
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in redacted), encoding="utf-8")
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
