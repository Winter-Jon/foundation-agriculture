#!/usr/bin/env python3
"""Promote only newly-valid Direct rejection records after a stricter-contract-compatible audit change."""
from __future__ import annotations

import argparse, json
from pathlib import Path

from agrinet.data.rebuild_sft import long_direct_audit_errors
from agrinet.data.sft_recovery import read_jsonl, write_jsonl
from tools.rag_distill.collect_hermes_1to1_v2 import answer_matches_target


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--rejected", type=Path, required=True); parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists(): raise RuntimeError(f"destination exists: {args.destination}")
    promoted=[]; retained=[]
    for record in read_jsonl(args.rejected):
        target=record.get("target") if isinstance(record.get("target"),dict) else {}
        content=str(record.get("generated") or "")
        row={"sample_id": target.get("target_id"), "images": [target.get("query_image")], "messages": [{"role":"user","content":"<image> Identify this."},{"role":"assistant","content":content}], "metadata":target}
        errors=long_direct_audit_errors(row,forbidden_hashes=set())
        if not answer_matches_target(target,content): errors.append("answer_target_mismatch")
        if not errors:
            row["metadata"]={**target,"re_audited_from":str(args.rejected),"re_audit_reason":"language_aware_zh_length_floor"}; promoted.append(row)
        else: retained.append({**record,"re_audit_errors":sorted(set(errors))})
    args.destination.mkdir(parents=True); write_jsonl(args.destination/"accepted.jsonl",promoted); write_jsonl(args.destination/"retained_rejected.jsonl",retained)
    (args.destination/"report.json").write_text(json.dumps({"source":str(args.rejected),"promoted":len(promoted),"retained":len(retained)},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print((args.destination/"report.json").read_text(encoding="utf-8"))


if __name__ == "__main__": main()
