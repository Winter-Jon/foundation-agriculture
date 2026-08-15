#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def read(path):
    with Path(path).open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--base",required=True); p.add_argument("--increment",action="append",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    rows=read(a.base); seen={r.get("sample_id") for r in rows}; duplicates=[]
    for source in a.increment:
        for row in read(source):
            sid=row.get("sample_id")
            if sid in seen: duplicates.append(sid); continue
            seen.add(sid); rows.append(row)
    if duplicates: raise SystemExit(f"duplicate sample IDs: {duplicates[:5]}")
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(json.dumps({"rows":len(rows),"base_rows":len(read(a.base)),"increment_rows":len(rows)-len(read(a.base)),"unique_ids":len(seen),"output":str(out)}))
if __name__=="__main__": main()
