#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from tools.rag_distill.schema import TOOL_NAME, validate_tool_arguments
except ModuleNotFoundError:
    from rag_distill.schema import TOOL_NAME, validate_tool_arguments

def read(path):
    with Path(path).open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]

def first_tool_errors(row):
    messages=row.get("messages") or []
    assistant=[m for m in messages if m.get("role")=="assistant"]
    errors=[]
    if not assistant: return ["missing_assistant"]
    first=assistant[0]; content=first.get("content")
    if not isinstance(content,str): return ["first_assistant_not_text_json"]
    try: call=json.loads(content)
    except Exception: return ["first_assistant_not_json_tool_call"]
    if not isinstance(call,dict) or call.get("name")!=TOOL_NAME or set(call)!={"name","arguments"}: return ["first_assistant_wrong_tool_shape"]
    errors.extend(validate_tool_arguments(call.get("arguments") or {}))
    if errors: return [f"first_call_{e}" for e in errors]
    return errors

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--accepted-output",required=True); p.add_argument("--rejected-output",required=True); p.add_argument("--report",required=True); a=p.parse_args()
    accepted=[]; rejected=[]
    for row in read(a.input):
        errors=first_tool_errors(row)
        if errors: rejected.append({"sample_id":row.get("sample_id"),"reasons":errors,"row":row})
        else: accepted.append(row)
    for path,data in ((a.accepted_output,accepted),(a.rejected_output,rejected)):
        out=Path(path); out.parent.mkdir(parents=True,exist_ok=True)
        with out.open("w",encoding="utf-8") as f:
            for row in data: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    report={"input":len(accepted)+len(rejected),"protocol_valid":len(accepted),"protocol_rejected":len(rejected),"rejection_reasons":{}}
    for item in rejected:
        for reason in item["reasons"]: report["rejection_reasons"][reason]=report["rejection_reasons"].get(reason,0)+1
    Path(a.report).parent.mkdir(parents=True,exist_ok=True); Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False))
    raise SystemExit(0 if not rejected else 1)
if __name__=="__main__": main()
