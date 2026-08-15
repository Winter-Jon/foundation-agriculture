#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import re
from .validate_semantic_quality import semantic_row_errors


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

def quality_score(row: dict[str, Any]) -> tuple[float, int]:
    metadata = row.get("metadata", {})
    aliases = {str(x).lower() for x in metadata.get("label_aliases", []) if x}
    best = 0.0
    for message in row.get("messages", []):
        if message.get("role") != "tool_response": continue
        response = json.loads(message.get("content", "{}"))
        for result in response.get("results", []):
            names = {str(result.get(k) or "").lower() for k in ("class_name", "chinese_name")}
            if aliases & names:
                best = max(best, float(result.get("score") or 0) + 1.0 / max(int(result.get("rank") or 99), 1))
    types = [str(value) for value in metadata.get("retrieval_types", [])]
    preferred = [str(value) for value in metadata.get("preferred_sequence", [])]
    turns = int(metadata.get("retrieval_turns") or 0)
    length_penalty = len(str(row.get("messages", [])).split()) / 10000
    route_bonus = 0.2 if preferred and types[: min(len(types), len(preferred))] == preferred[: min(len(types), len(preferred))] else 0.0
    completion_bonus = 0.1 if preferred and types == preferred else 0.0
    name_bonus = 0.1 if types and types[-1] == "name" and "name" in preferred else 0.0
    repeat_penalty = 0.1 * sum(left == right for left, right in zip(types, types[1:]))
    score = best + route_bonus + completion_bonus + name_bonus - repeat_penalty - 0.05 * max(turns - len(preferred or types), 0) - length_penalty
    return score, -int(metadata.get("candidate_index") or 99)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", required=True)
    args = parser.parse_args()
    pilot = Path(args.pilot_dir)
    batch = pilot / "candidates" / "batch"
    accepted = read_jsonl(batch / "train" / "agent_sft.accepted.jsonl")
    rejected = read_jsonl(batch / "traces" / "rejected_trajectories.jsonl")
    for supplement in sorted((pilot / "candidates").glob("supplement-*")):
        accepted.extend(read_jsonl(supplement / "train" / "agent_sft.accepted.jsonl"))
        rejected.extend(read_jsonl(supplement / "traces" / "rejected_trajectories.jsonl"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    semantic_rejected=[]
    for row in accepted:
        reasons=semantic_row_errors(row)
        if reasons:
            semantic_rejected.append({"sample_id":row.get("sample_id"),"target_id":row.get("metadata",{}).get("target_id"),"reasons":[f"semantic_{reason}" for reason in reasons]})
        else: grouped[str(row.get("metadata", {}).get("target_id"))].append(row)
    target_best = []
    not_selected = []
    for target_id, rows in sorted(grouped.items()):
        rows.sort(key=quality_score, reverse=True)
        target_best.append(rows[0])
        not_selected.extend({"sample_id": row.get("sample_id"), "target_id": target_id,
                             "reasons": ["valid_but_not_selected"], "candidate_index": row.get("metadata", {}).get("candidate_index")}
                            for row in rows[1:])
    cells: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in target_best:
        meta = row["metadata"]
        cells[(meta["trajectory_mode"], meta["question_type"], meta["language"], meta["task_domain"])].append(row)
    selected = []
    cell_shortfalls = {}
    used_pre_rag=set()
    for cell, rows in sorted(cells.items()):
        rows.sort(key=quality_score, reverse=True)
        if cell[1] == "option":
            chosen = []
            local_pre = set(used_pre_rag)
            used_strategies=set()
            for letter in "ABCD":
                match = next((row for row in rows if row["metadata"].get("correct_option") == letter and row["metadata"].get("strategy_id") not in used_strategies and next((m["content"] for m in row["messages"] if m["role"]=="assistant"),"") not in local_pre), None)
                if match is None:
                    match = next((row for row in rows if row["metadata"].get("correct_option") == letter and next((m["content"] for m in row["messages"] if m["role"]=="assistant"),"") not in local_pre), None)
                if match is not None:
                    chosen.append(match)
                    used_strategies.add(match["metadata"].get("strategy_id"))
                    local_pre.add(next((m["content"] for m in match["messages"] if m["role"]=="assistant"),""))
        else:
            chosen=[]
            local_pre=set(used_pre_rag)
            used_strategies=set()
            for require_new_strategy in (True, False):
                for row in rows:
                    strategy=row["metadata"].get("strategy_id")
                    pre=next((m["content"] for m in row["messages"] if m["role"]=="assistant"),"")
                    if row in chosen or pre in local_pre or (require_new_strategy and strategy in used_strategies):
                        continue
                    chosen.append(row); local_pre.add(pre); used_strategies.add(strategy)
                    if len(chosen)==4: break
                if len(chosen)==4: break
        selected.extend(chosen)
        used_pre_rag.update(next((m["content"] for m in row["messages"] if m["role"]=="assistant"),"") for row in chosen)
        if len(chosen) < 4: cell_shortfalls["/".join(cell)] = 4 - len(chosen)
        chosen_ids = {id(row) for row in chosen}
        not_selected.extend({"sample_id": row.get("sample_id"), "target_id": row["metadata"].get("target_id"),
                             "reasons": ["valid_reserve_not_selected"], "candidate_index": row["metadata"].get("candidate_index")}
                            for row in rows if id(row) not in chosen_ids)
    write_jsonl(pilot / "accepted" / "rag_selected.jsonl", selected)
    write_jsonl(pilot / "rejected" / "rag_rejected.jsonl", rejected + semantic_rejected + not_selected)
    direct = read_jsonl(pilot / "accepted" / "direct_open.jsonl") + read_jsonl(pilot / "accepted" / "direct_option.jsonl")
    write_jsonl(pilot / "accepted" / "pilot_train_review.jsonl", direct + selected)
    target_count = len(read_jsonl(pilot / "plan" / "rag_targets.jsonl"))
    query_images = [str(row.get("images", [""])[0]) for row in direct + selected]
    duplicate_query_images = len(query_images) - len(set(query_images))
    summary = {
        "rag_targets_planned": target_count, "rag_selected": len(selected),
        "rag_quota_shortfalls": cell_shortfalls,
        "direct_selected": len(direct), "pilot_train_review_rows": len(direct) + len(selected),
        "selection_policy": "target evidence quality, strategy conformance and coverage, concision; candidate_index tie-break",
        "strategy_distribution": dict(Counter(row.get("metadata", {}).get("strategy_id", "legacy_visual") for row in selected)),
        "duplicate_query_images": duplicate_query_images,
        "rejection_reasons": dict(Counter(reason for row in rejected + semantic_rejected + not_selected for reason in row.get("reasons", []))),
        "training_authorized": False,
    }
    (pilot / "review" / "generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if len(selected) == 48 and not cell_shortfalls and duplicate_query_images == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
