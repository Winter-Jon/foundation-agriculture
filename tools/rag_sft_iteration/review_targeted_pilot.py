#!/usr/bin/env python3
"""Create a compact review package for a completed targeted pilot."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

def rows(path: Path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--artifact",required=True); args=ap.parse_args()
    root=Path(args.artifact); accepted=rows(root/"train/agent_sft.accepted.jsonl"); rejected=rows(root/"traces/rejected_trajectories.jsonl")
    route=Counter((r.get("metadata",{}).get("generation_route") or "unknown") for r in accepted)
    langs=Counter((r.get("metadata",{}).get("language") or "unknown") for r in accepted)
    tool_ok=0; tool_total=0; protocol_errors=[]
    for row in accepted:
        messages=row.get("messages") or []
        tool_responses=[m for m in messages if m.get("role")=="tool_response"]
        tool_total += len(tool_responses)
        for m in tool_responses:
            try:
                payload=json.loads(m.get("content") or "{}")
                if payload.get("status")=="success": tool_ok += 1
            except Exception: protocol_errors.append(row.get("metadata",{}).get("sample_id"))
    summary={"artifact":str(root),"accepted":len(accepted),"rejected":len(rejected),"routes":dict(route),"languages":dict(langs),"tool_responses":tool_total,"tool_success":tool_ok,"protocol_parse_errors":protocol_errors,"review_ready":bool(accepted) and not protocol_errors and tool_ok==tool_total}
    (root/"reports").mkdir(parents=True,exist_ok=True)
    (root/"reports/targeted_review.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))
    return 0 if summary["review_ready"] else 2
if __name__=="__main__": raise SystemExit(main())
