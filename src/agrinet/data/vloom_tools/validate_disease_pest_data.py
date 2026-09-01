import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REQUIRED_RESULT_FIELDS = {
    "task_domain",
    "final_label",
    "final_label_zh",
    "candidate_labels",
    "query_visual_evidence",
    "positive_reference_alignment",
    "negative_reference_contrast",
    "wiki_evidence",
    "agricultural_interpretation",
    "uncertainty",
    "answer_check",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
    return rows


def code_domain(code: str) -> str:
    if code.startswith("N04"):
        return "disease"
    if code.startswith("N05"):
        return "pest"
    return "unknown"


def add_error(errors: List[str], message: str) -> None:
    if len(errors) < 50:
        errors.append(message)


def validate_classes(rows: Sequence[Dict[str, Any]], expected_count: int, domain: str, errors: List[str]) -> None:
    if len(rows) != expected_count:
        add_error(errors, f"{domain}: expected {expected_count} classes, found {len(rows)}")
    for row in rows:
        code = str(row.get("code", ""))
        if code_domain(code) != domain:
            add_error(errors, f"{domain}: unexpected class code {code}")
        if not row.get("wiki_available"):
            add_error(errors, f"{code}: wiki_available is false")
        evidence = row.get("wiki_evidence") or []
        if not evidence or evidence == ["insufficient wiki knowledge"]:
            add_error(errors, f"{code}: missing usable wiki evidence")


def validate_pairs(rows: Sequence[Dict[str, Any]], errors: List[str], top_k: int) -> None:
    for row in rows:
        anchor = str(row.get("anchor") or row.get("label") or "")
        domain = str(row.get("task_domain") or code_domain(anchor))
        negatives = row.get("hard_negatives") or []
        if len(negatives) < top_k:
            add_error(errors, f"{anchor}: expected at least {top_k} hard negatives, found {len(negatives)}")
        for neg in negatives:
            neg_code = str(neg.get("code", ""))
            if code_domain(neg_code) != domain:
                add_error(errors, f"{anchor}: cross-domain hard negative {neg_code}")
            if neg_code == anchor:
                add_error(errors, f"{anchor}: self hard negative")


def validate_samples(rows: Sequence[Dict[str, Any]], errors: List[str], expected_negatives: int) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        label = str(row.get("final_label", ""))
        domain = str(row.get("task_domain") or code_domain(label))
        counts[domain] += 1

        query = row.get("query_image")
        positives = row.get("positive_reference_images") or []
        negatives = row.get("negative_reference_images") or []
        candidates = row.get("candidate_labels") or []
        candidate_codes = [str(item.get("code", item)) for item in candidates]

        if not query or not Path(query).is_file():
            add_error(errors, f"{sample_id}: query image does not exist: {query}")
        if len(positives) != 2:
            add_error(errors, f"{sample_id}: expected 2 positive refs, found {len(positives)}")
        if len(negatives) != expected_negatives:
            add_error(errors, f"{sample_id}: expected {expected_negatives} negative refs, found {len(negatives)}")
        expected_candidates = expected_negatives + 1
        if len(candidates) != expected_candidates:
            add_error(errors, f"{sample_id}: expected {expected_candidates} candidates, found {len(candidates)}")
        if label not in candidate_codes:
            add_error(errors, f"{sample_id}: final_label {label} missing from candidates")
        if query in positives:
            add_error(errors, f"{sample_id}: query reused as positive reference")
        if len(set(positives)) != len(positives):
            add_error(errors, f"{sample_id}: duplicate positive references")

        for path in positives:
            if not Path(path).is_file():
                add_error(errors, f"{sample_id}: positive ref does not exist: {path}")
        for neg in negatives:
            neg_code = str(neg.get("code", ""))
            neg_path = neg.get("image_path")
            if code_domain(neg_code) != domain:
                add_error(errors, f"{sample_id}: cross-domain negative {neg_code}")
            if neg_code == label:
                add_error(errors, f"{sample_id}: negative repeats final label")
            if not neg_path or not Path(neg_path).is_file():
                add_error(errors, f"{sample_id}: negative ref does not exist: {neg_path}")
            if neg_path == query:
                add_error(errors, f"{sample_id}: query reused as negative reference")
    return counts


def validate_results(path: Path, errors: List[str]) -> Dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = {"records": len(payload), "parsed": 0, "missing_fields": 0, "label_outside_candidates": 0}
    for key, row in payload.items():
        if not row or not row.get("result"):
            add_error(errors, f"{key}: missing parsed JSON result")
            continue
        result = row["result"]
        stats["parsed"] += 1
        missing = sorted(REQUIRED_RESULT_FIELDS - set(result))
        if missing:
            stats["missing_fields"] += 1
            add_error(errors, f"{key}: result missing fields {missing}")
        final_label = result.get("final_label")
        candidates = result.get("candidate_labels") or []
        if final_label not in candidates:
            stats["label_outside_candidates"] += 1
            add_error(errors, f"{key}: final_label {final_label} outside candidate_labels")
    return stats


def print_summary(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AgriNet disease/pest contrast-CoT manifests and results.")
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/vlm_data/disease_pest"))
    parser.add_argument("--results", type=Path, help="Optional VLooM task result JSON to validate.")
    parser.add_argument("--expected-disease", type=int, default=145)
    parser.add_argument("--expected-pest", type=int, default=72)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--expected-negatives", type=int, default=3)
    parser.add_argument("--samples-per-class", type=int, default=1)
    args = parser.parse_args()

    errors: List[str] = []
    disease_rows = read_jsonl(args.data_dir / "classes_all_disease.jsonl")
    pest_rows = read_jsonl(args.data_dir / "classes_all_pest.jsonl")
    pair_rows = read_jsonl(args.data_dir / "finegrained_pairs_vit_base.jsonl")
    sample_rows = read_jsonl(args.data_dir / "contrast_samples_vit_base.jsonl")

    validate_classes(disease_rows, args.expected_disease, "disease", errors)
    validate_classes(pest_rows, args.expected_pest, "pest", errors)
    validate_pairs(pair_rows, errors, args.top_k)
    sample_counts = validate_samples(sample_rows, errors, args.expected_negatives)

    expected_samples = (args.expected_disease + args.expected_pest) * args.samples_per_class
    if len(sample_rows) != expected_samples:
        add_error(errors, f"samples: expected {expected_samples}, found {len(sample_rows)}")

    summary = [
        f"disease_classes={len(disease_rows)}",
        f"pest_classes={len(pest_rows)}",
        f"samples={len(sample_rows)}",
        f"sample_domains={dict(sample_counts)}",
    ]
    if args.results:
        result_stats = validate_results(args.results, errors)
        summary.extend(f"results_{key}={value}" for key, value in result_stats.items())

    summary.append(f"errors={len(errors)}")
    print_summary(summary)
    if errors:
        print("first_errors:")
        print_summary(f"- {error}" for error in errors)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
