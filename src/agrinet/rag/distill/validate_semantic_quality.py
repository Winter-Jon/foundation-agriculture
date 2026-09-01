#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

FORBIDDEN = (
    "Knowledge Verification", "Wiki Knowledge", "知识验证", "论文摘要",
    "分类谱系", "<title>摘要</title>", "downstream normalization", "private target",
)

def semantic_row_errors(row: dict) -> list[str]:
    messages = row.get("messages") or []; final = str(messages[-1].get("content", "")) if messages else ""; errors=[]
    if len(final) > 1200: errors.append(f"reasoning_too_long:{len(final)}")
    if any(marker in final for marker in FORBIDDEN): errors.append("forbidden_knowledge_or_development_text")
    if final.count("Visual Observation:") + final.count("视觉观察：") > 1: errors.append("duplicate_stage_heading")
    if row.get("tools"):
        first = next((str(msg.get("content", "")) for msg in messages if msg.get("role") == "assistant"), "")
        if not any(marker in first for marker in ("Candidate Analysis:", "候选分析：")): errors.append("missing_pre_rag_candidates")
        if "视觉观察：" in first:
            observation=first.split("视觉观察：",1)[1].split("候选分析：",1)[0]
            if "、" not in observation: errors.append("fewer_than_two_visible_traits")
        elif "Visual Observation:" in first:
            observation=first.split("Visual Observation:",1)[1].split("Candidate Analysis:",1)[0]
            if not any(separator in observation.lower() for separator in (",", " and ", " with ", " on ", " perched ", " clustered ")):
                errors.append("fewer_than_two_visible_traits")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    data = read_rows(args.train_file)
    errors: list[str] = []
    pre_rag: Counter[str] = Counter()
    image_routes: list[tuple[str, str]] = []
    for row in data:
        sample_id = str(row.get("sample_id") or "<missing>")
        messages = row.get("messages") or []
        final = str(messages[-1].get("content", "")) if messages else ""
        route = str((row.get("metadata") or {}).get("generation_route") or "legacy")
        image_routes.append((str((row.get("images") or [""])[0]), route))
        errors.extend(f"{sample_id}: {reason}" for reason in semantic_row_errors(row))
        if row.get("tools"):
            first = next((str(msg.get("content", "")) for msg in messages if msg.get("role") == "assistant"), "")
            pre_rag[first] += 1
    errors.extend(f"repeated_pre_rag_text:{count}" for text, count in pre_rag.items() if text and count > 1)
    # Oracle and Blind are intentional paired views of the same query image.
    # Only duplicate images within one generation route indicate resampling.
    duplicate_images = len(image_routes) - len(set(image_routes))
    if duplicate_images:
        errors.append(f"duplicate_query_images:{duplicate_images}")
    report = {
        "rows": len(data), "protocol_valid": None, "semantic_valid": not errors,
        "error_count": len(errors), "errors": errors,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
