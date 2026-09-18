"""Merge a quota-selected collection with independent supplement trajectories."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agrinet.rag.e344_pipeline import write, write_rows
from agrinet.rag.e344_prepare import IDENTITIES, PRIORITY, digest, rows
from agrinet.rag.full_collection_acceptance import CHECKS


def _load_eligible(root: Path):
    """Load template-passing rows, retaining their matching provenance."""
    data = list(rows(root / "data.jsonl"))
    lineage = list(rows(root / "lineage.jsonl"))
    report = json.loads((root / "template-check.json").read_text())
    if report.get("data_sha256") != digest(root / "data.jsonl"):
        raise ValueError(f"template_input_hash_mismatch:{root}")
    checks = report.get("rows", [])
    if len(data) != len(lineage) or len(data) != len(checks):
        raise ValueError(f"incomplete_template_or_lineage:{root}")
    if [check.get("row_index") for check in checks] != list(range(len(data))):
        raise ValueError(f"template_row_order_mismatch:{root}")
    eligible, excluded = [], []
    for row, trace, check in zip(data, lineage, checks, strict=True):
        source = trace.get("source", {})
        if not all(check.get(key, False) for key in CHECKS):
            excluded.append({"sample_id": source.get("sample_id"),
                             "reason": check.get("error", "template_check_failed")})
        else:
            eligible.append((row, trace))
    return eligible, excluded, report


def _key(trace):
    source = trace["source"]
    return source["canonical_class_code"], source["arm"], source["question_type"]


def accept(first_root, supplement_roots, output_root):
    """Preserve first-round quota rows then fill its documented deficits."""
    first_root, output_root = map(Path, (first_root, output_root))
    supplement_roots = [Path(root) for root in supplement_roots]
    output_root.mkdir(parents=True, exist_ok=True)
    first_eligible, first_exclusions, first_template = _load_eligible(first_root)
    supplements = [(root, *_load_eligible(root)) for root in supplement_roots]

    first_data = list(rows(first_root / "quota-data.jsonl"))
    first_lineage = list(rows(first_root / "quota-lineage.jsonl"))
    if len(first_data) != len(first_lineage):
        raise ValueError("first_quota_data_lineage_mismatch")
    first_pairs = list(zip(first_data, first_lineage, strict=True))
    eligible_ids = {trace["source"]["sample_id"] for _, trace in first_eligible}
    if any(trace["source"].get("sample_id") not in eligible_ids for _, trace in first_pairs):
        raise ValueError("first_quota_contains_template_rejected_row")

    deficits = json.loads((first_root / "quota-deficits.json").read_text())
    targets = {(row["canonical_class_code"], row["arm"], row["question_type"]): row["target"]
               for row in deficits}
    expected_targets = {(code, arm, question): quota
                        for code, _, _ in targets for arm, question, quota in PRIORITY}
    if targets != expected_targets:
        raise ValueError("first_quota_target_contract_mismatch")

    selected, selected_lineage, exclusions = [], [], list(first_exclusions)
    used = {identity: set() for identity in IDENTITIES}
    counts = Counter()

    def add(row, trace, origin):
        source = trace["source"]
        missing = [identity for identity in IDENTITIES if not source.get(identity)]
        if missing:
            raise ValueError(f"missing_identity:{source.get('sample_id')}:{','.join(missing)}")
        if any(source[identity] in used[identity] for identity in IDENTITIES):
            raise ValueError(f"accepted_identity_collision:{source.get('sample_id')}")
        for identity in IDENTITIES:
            used[identity].add(source[identity])
        selected.append(row)
        selected_lineage.append({**trace, "collection_origin": origin})
        counts[_key(trace)] += 1

    for row, trace in first_pairs:
        key = _key(trace)
        if key not in targets or counts[key] >= targets[key]:
            raise ValueError(f"invalid_first_quota_row:{trace['source'].get('sample_id')}")
        add(row, trace, "first_round")

    for index, (_, supplement_eligible, supplement_exclusions, _) in enumerate(supplements, 1):
        for row, trace in supplement_eligible:
            key = _key(trace)
            source = trace["source"]
            if key not in targets or counts[key] >= targets[key]:
                exclusions.append({"sample_id": source.get("sample_id"), "reason": "quota_already_filled"})
                continue
            if any(source.get(identity) in used[identity] for identity in IDENTITIES):
                exclusions.append({"sample_id": source.get("sample_id"), "reason": "cross_round_identity_collision"})
                continue
            add(row, trace, f"supplement_{index}")
        exclusions.extend(supplement_exclusions)

    final_deficits = [{"canonical_class_code": code, "arm": arm, "question_type": question,
                       "target": target, "available": counts[code, arm, question],
                       "deficit": max(0, target - counts[code, arm, question])}
                      for (code, arm, question), target in sorted(targets.items())]
    write_rows(output_root / "data.jsonl", selected)
    write_rows(output_root / "lineage.jsonl", selected_lineage)
    write(output_root / "deficits.json", final_deficits)
    write(output_root / "exclusions.json", exclusions)
    summary = {"target": sum(targets.values()), "accepted": len(selected),
               "deficit": sum(item["deficit"] for item in final_deficits),
               "first_round_accepted": len(first_pairs),
               "supplement_accepted": len(selected) - len(first_pairs),
               "template_accepted": {"first_round": len(first_eligible),
                                     "supplements": [len(eligible) for _, eligible, _, _ in supplements]},
               "input_sha256": {"first_data": digest(first_root / "data.jsonl"),
                                "first_template": digest(first_root / "template-check.json"),
                                "supplements": [{"data": digest(root / "data.jsonl"),
                                                   "template": digest(root / "template-check.json")}
                                                  for root, _, _, _ in supplements]}}
    write(output_root / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-root", required=True)
    parser.add_argument("--supplement-root", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    arguments = parser.parse_args()
    print(json.dumps(accept(arguments.first_root, arguments.supplement_root, arguments.output_root)))
