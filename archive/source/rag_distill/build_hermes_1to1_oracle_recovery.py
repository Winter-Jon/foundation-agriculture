#!/usr/bin/env python3
"""Build one lineaged, capacity-checked Oracle recovery manifest from Blind rejects."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.data.rebuild_sft import canonical_json_hash, cell_of, load_hermes_1to1_specification
from agrinet.data.sft_recovery import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cap-to-remaining", action="store_true", help="Select only the deterministic prefix that fits each remaining cell capacity.")
    args = parser.parse_args()

    rejected = read_jsonl(args.rejected)
    targets: list[dict[str, Any]] = []
    for row in rejected:
        target = row.get("target") if isinstance(row.get("target"), dict) else {}
        errors = row.get("errors") if isinstance(row.get("errors"), list) else []
        if errors != ["answer_target_mismatch"]:
            continue
        if target.get("generation_route") != "blind_evidence" or target.get("label_visible_to_teacher") is not False:
            raise ValueError(f"not an eligible Blind mismatch: {target.get('target_id')}")
        targets.append({**target, "oracle_recovery_of": {"source_rejected": str(args.rejected), "reason": "answer_target_mismatch"}})
    if not targets:
        raise ValueError("no eligible Blind answer-target mismatches")
    hashes = [str(target.get("image_sha256") or "") for target in targets]
    if len(hashes) != len(set(hashes)) or "" in hashes:
        raise ValueError("Oracle recovery targets must have distinct known image hashes")

    existing: Counter[tuple[str, str, str]] = Counter()
    for path in args.collection_root.glob("*/accepted.jsonl"):
        for row in read_jsonl(path):
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            if metadata.get("generation_route") == "oracle_grounded":
                existing[cell_of(row)] += 1
    cap = int(load_hermes_1to1_specification()["oracle_rag_cap_per_cell"])
    if args.cap_to_remaining:
        selected: list[dict[str, Any]] = []
        selected_counts: Counter[tuple[str, str, str]] = Counter()
        for target in sorted(targets, key=lambda item: str(item.get("target_id") or "")):
            cell = cell_of(target)
            if existing[cell] + selected_counts[cell] < cap:
                selected.append(target); selected_counts[cell] += 1
        targets, requested = selected, selected_counts
    else:
        requested = Counter(cell_of(target) for target in targets)
    over = [f"{'/'.join(cell)}:{existing[cell]}+{count}>{cap}" for cell, count in sorted(requested.items()) if existing[cell] + count > cap]
    if over:
        raise ValueError("oracle capacity exceeded: " + "; ".join(over))
    if not targets:
        raise ValueError("no Oracle capacity remains for eligible targets")

    report = {
        "schema_version": "agrinet.hermes-1to1-oracle-recovery/v1",
        "targets": len(targets),
        "source_rejected": str(args.rejected),
        "oracle_before": {"/".join(cell): existing[cell] for cell in sorted(requested)},
        "oracle_requested": {"/".join(cell): requested[cell] for cell in sorted(requested)},
        "oracle_after": {"/".join(cell): existing[cell] + requested[cell] for cell in sorted(requested)},
        "oracle_cap": cap,
        "targets_sha256": canonical_json_hash(targets),
        "training_authorized": False,
    }
    args.destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.destination / "rag_targets.jsonl", targets)
    (args.destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
