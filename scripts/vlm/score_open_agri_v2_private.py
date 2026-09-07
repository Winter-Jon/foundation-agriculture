#!/usr/bin/env python3
"""Offline-only bilingual-canonical OpenAgri v2 scorer.

For a given private class code, either frozen canonical surface form (English or
Chinese) is accepted.  This keeps language-conditioned inference comparable
when a model correctly identifies a class but returns the other benchmark
language.  Private labels and codes never leave this process.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG = REPO_ROOT / "outputs/artifacts/datasets/m1-direct-current-hcv-v1/classes.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scorer_module() -> Any:
    path = REPO_ROOT / "vlm/eval/tools/normalize_answers.py"
    spec = importlib.util.spec_from_file_location("open_agri_answer_normalizer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registry_answers(
    path: Path | None, catalog: dict[str, dict[str, Any]], normalizer: Any
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Load an optional public answer registry for a derived SFT contract.

    The default scorer is intentionally unchanged.  A registry supplements the
    two catalog surface forms only for its matching code, so it resolves public
    display-name collisions without allowing cross-class aliases.
    """
    accepted: dict[str, set[str]] = defaultdict(set)
    canonical_by_source: dict[str, str] = {}
    if path is None:
        return accepted, canonical_by_source
    for row in load_jsonl(path):
        code = str(row.get("code") or "")
        if code not in catalog:
            raise ValueError("canonical registry contains a code outside the frozen private catalog")
        source_codes = [str(value) for value in row.get("source_codes") or [code]]
        if not source_codes or code not in source_codes:
            raise ValueError("canonical registry must include its canonical code in source_codes")
        for source_code in source_codes:
            if source_code not in catalog:
                raise ValueError("canonical registry source code is outside the frozen private catalog")
            if source_code in canonical_by_source:
                raise ValueError("canonical registry maps a source code more than once")
            canonical_by_source[source_code] = code
        for key in ("canonical_en", "canonical_zh", "english_aliases", "chinese_aliases"):
            values = row.get(key)
            values = values if isinstance(values, list) else [values]
            for value in values:
                answer = normalizer.normalize_text(value)
                if not answer:
                    raise ValueError(f"canonical registry has an empty {key}")
                accepted[code].add(answer)
    if set(canonical_by_source) != set(catalog):
        raise ValueError("canonical registry must cover exactly the frozen private catalog source codes")
    return accepted, canonical_by_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True, help="Private dev_truth.jsonl or test_truth.jsonl.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Redacted per-row scores without labels/codes.")
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument(
        "--canonical-registry", type=Path,
        help="Optional public canonical_name_registry.jsonl from a derived SFT artifact.",
    )
    args = parser.parse_args()
    predictions = load_jsonl(args.predictions)
    truth = load_jsonl(args.truth)
    catalog = {str(row["code"]): row for row in load_jsonl(CATALOG)}
    truth_by_id = {str(row["id"]): row for row in truth}
    ids = [str(row.get("id") or "") for row in predictions]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(truth_by_id):
        raise ValueError("prediction IDs must be a one-to-one match with the private truth file")
    normalizer = scorer_module()
    registry, canonical_by_source = registry_answers(args.canonical_registry, catalog, normalizer)
    results: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in predictions:
        item_id = str(row["id"])
        private = truth_by_id[item_id]
        source_code = str(private["class_code"])
        code = canonical_by_source.get(source_code, source_code)
        class_row = catalog.get(code)
        if class_row is None:
            raise ValueError("private truth code missing from frozen class catalog")
        final, source = normalizer.extract_final_answer(str(row.get("prediction") or ""))
        answer = normalizer.normalize_text(final)
        accepted_answers = {
            normalizer.normalize_text(class_row["english_name"]),
            normalizer.normalize_text(class_row["chinese_name"]),
        }
        accepted_answers.update(registry.get(code, set()))
        accepted_answers.discard("")
        correct = bool(answer and answer in accepted_answers and not row.get("error"))
        key = (str(row.get("language") or ""), str(private.get("evaluation_bucket") or ""))
        groups[key]["rows"] += 1
        groups[key]["correct"] += int(correct)
        groups[key]["errors"] += int(bool(row.get("error")))
        results.append({
            "id": item_id, "language": row.get("language"), "route": row.get("route"),
            "prediction": row.get("prediction"), "final_answer_text": final,
            "answer_extraction_source": source, "correct": correct,
            "protocol_error": bool(row.get("error")) or bool((row.get("protocol") or {}).get("has_protocol_event")),
        })
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    total = Counter()
    summary_groups = {}
    for key, values in sorted(groups.items()):
        total.update(values)
        summary_groups["/".join(key)] = {
            "rows": values["rows"], "accuracy": values["correct"] / values["rows"],
            "request_error_rate": values["errors"] / values["rows"],
        }
    metrics = {
        "schema_version": "agrinet.open-agri-v2-private-score/v4-bilingual-canonical-registry",
        "private_truth_used_offline_only": True,
        "private_labels_or_codes_emitted": False,
        "accepted_answer_languages": ["en", "zh"],
        "canonical_registry": str(args.canonical_registry) if args.canonical_registry else None,
        "rows": total["rows"], "accuracy": total["correct"] / total["rows"],
        "request_error_rate": total["errors"] / total["rows"], "by_language_and_bucket": summary_groups,
    }
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
