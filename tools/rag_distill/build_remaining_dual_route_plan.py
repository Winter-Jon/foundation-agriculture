from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build non-overlapping Oracle/Blind attempts for remaining targets.")
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--pilot-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    completed_targets = {row["target_id"] for row in read_jsonl(args.pilot_plan)}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(args.attempts):
        if row["target_id"] not in completed_targets:
            grouped[row["target_id"]].append(row)

    selected: list[dict] = []
    strategy_counts: Counter[str] = Counter()
    route_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for target_id in sorted(grouped):
        candidates = sorted(grouped[target_id], key=lambda row: row["candidate_index"])
        used: set[str] = set()
        for route in ("oracle_grounded", "blind_evidence"):
            choices = [row for row in candidates if row["strategy_id"] not in used] or candidates
            source = min(
                choices,
                key=lambda row: (strategy_counts[row["strategy_id"]], route_counts[route][row["strategy_id"]], row["candidate_index"]),
            )
            index = len(selected) + 1
            selected.append(
                {
                    **source,
                    "generation_route": route,
                    "route_attempt_index": 1,
                    "sample_id": f"remaining-{index:03d}-{route}",
                }
            )
            used.add(source["strategy_id"])
            strategy_counts[source["strategy_id"]] += 1
            route_counts[route][source["strategy_id"]] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "attempts": len(selected),
        "targets": len(grouped),
        "excluded_pilot_targets": len(completed_targets),
        "routes": Counter(row["generation_route"] for row in selected),
        "strategies": strategy_counts,
    }
    print(json.dumps(summary, ensure_ascii=False, default=dict))
    return int(len(selected) != 2 * len(grouped) or any(len(rows) < 2 for rows in grouped.values()))


if __name__ == "__main__":
    raise SystemExit(main())
