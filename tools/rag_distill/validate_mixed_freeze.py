#!/usr/bin/env python3
"""Validate a mixed freeze with route-specific validators and one summary gate."""
from __future__ import annotations
import argparse, json, re, subprocess
from collections import Counter
from pathlib import Path
from typing import Any

ZH = re.compile(r"[\u3400-\u9fff]")

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]

def image_key(row: dict[str, Any]) -> str:
    return str((row.get("images") or [""])[0]).replace("\\", "/")

def direct_errors(row: dict[str, Any], idx: int) -> list[str]:
    sid = str(row.get("sample_id") or f"direct-{idx}")
    errors: list[str] = []
    msgs, images = row.get("messages"), row.get("images")
    if not isinstance(msgs, list) or not isinstance(images, list) or len(images) != 1: errors.append("direct_image_or_messages_invalid")
    if isinstance(msgs, list):
        roles = [m.get("role") for m in msgs if isinstance(m, dict)]
        if "user" not in roles or "assistant" not in roles: errors.append("direct_missing_user_or_assistant")
        if any(m.get("role") in {"tool_call", "tool_response"} for m in msgs if isinstance(m, dict)): errors.append("direct_contains_tool_message")
        answer = "\n".join(str(m.get("content", "")) for m in msgs if m.get("role") == "assistant")
        if "<answer>" not in answer or "</answer>" not in answer: errors.append("direct_answer_tags_missing")
        user = "\n".join(str(m.get("content", "")) for m in msgs if m.get("role") == "user")
        lang = "zh" if ZH.search(user) else "en"
        if lang == "en" and ZH.search(answer): errors.append("direct_language_mismatch_en")
        if lang == "zh" and not ZH.search(answer): errors.append("direct_language_mismatch_zh")
    else: lang = "unknown"
    image = image_key(row)
    domain = "pest" if "/N050" in image else "disease" if "/N040" in image else "unknown"
    return [f"{sid}:{e}" for e in errors]

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--artifact-dir", type=Path, required=True); ap.add_argument("--rag-file", type=Path, required=True); ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args(); mixed = read_jsonl(args.artifact_dir / "data.jsonl"); rag = read_jsonl(args.rag_file); direct = mixed[len(rag):]
    derrors = [e for i, r in enumerate(direct) for e in direct_errors(r, i)]
    images = [image_key(r) for r in mixed]
    stats = json.loads((args.artifact_dir / "statistics.json").read_text())
    report = {"artifact_dir":str(args.artifact_dir), "rows":len(mixed), "rag_rows":len(rag), "direct_rows":len(direct), "mixed_unique_images":len(set(images)), "direct_errors":derrors, "direct_error_count":len(derrors), "direct_coverage":stats.get("direct_coverage", {}), "rag_validation":"run separately with validate_artifact on the 48-row RAG artifact", "mixed_rag_domain_validator_misuse_avoided":True, "training_authorized":False, "formal_eval_authorized":False, "valid_structure":len(mixed)==80 and len(set(images))==80 and len(derrors)==0}
    args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report":str(args.report), "valid_structure":report["valid_structure"], "direct_error_count":len(derrors), "rows":len(mixed)}, ensure_ascii=False)); return 0 if report["valid_structure"] else 1

if __name__ == "__main__": raise SystemExit(main())
