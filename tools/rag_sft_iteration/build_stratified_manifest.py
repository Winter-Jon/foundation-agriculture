#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--per-cell',type=int,default=4); ap.add_argument('--total',type=int,default=96); a=ap.parse_args()
    cells=defaultdict(list)
    for line in Path(a.input).open(encoding='utf-8'):
        if line.strip():
            row=json.loads(line); key=(row.get('known_bucket'),row.get('task_domain'),row.get('language'),row.get('question_type')); cells[key].append(row)
    missing={key:len(rows) for key,rows in cells.items() if len(rows)<a.per_cell}
    expected=[(k,d,l,q) for k in ('known','unknown') for d in ('disease','pest') for l in ('en','zh') for q in ('open','option')]
    if any(key not in cells or len(cells[key])<a.per_cell for key in expected): raise SystemExit(f'insufficient cells: {missing}')
    selected=[]
    for key in expected: selected.extend(cells[key][:a.per_cell])
    remaining_by_key={key:list(cells[key][a.per_cell:]) for key in expected}
    extra=max(0, a.total-len(selected))
    while extra:
        progressed=False
        for key in expected:
            if remaining_by_key[key] and extra:
                selected.append(remaining_by_key[key].pop(0)); extra-=1; progressed=True
        if not progressed: break
    if len(selected) != a.total: raise SystemExit(f'insufficient rows for requested total: {len(selected)}')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for row in selected: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({'rows':len(selected),'cells':len(expected),'per_cell_min':a.per_cell,'output':str(out)},ensure_ascii=False))
if __name__=='__main__': main()
