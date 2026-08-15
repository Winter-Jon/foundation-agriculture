#!/usr/bin/env python3
"""Select a deterministic 16-row language/domain/open-option diagnostic."""
import argparse, json
from collections import defaultdict
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    rows=[json.loads(x) for x in args.source.open(encoding='utf-8') if x.strip()]
    cells=defaultdict(list)
    for r in rows:
        cells[(r.get('language','en'),r.get('task_domain','unknown'),r.get('question_type','open'),r.get('known_bucket','known'))].append(r)
    # Keep known/unknown and Open/Option visible in the diagnostic: two rows per
    # language x domain x question type, preferring known then unknown.
    selected=[]
    for lang in ('en','zh'):
      for dom in ('disease','pest'):
       for q in ('open','option'):
        pool=[]
        for bucket in ('known','unknown'):
          pool += sorted(cells[(lang,dom,q,bucket)], key=lambda x:str(x.get('id') or x.get('source_sample_id')))
        if len(pool)<2: raise SystemExit(f'cell {(lang,dom,q)} has {len(pool)} rows')
        selected += pool[:2]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w',encoding='utf-8') as f:
      for r in selected: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps({'rows':len(selected),'out':str(args.out),'cells':{ '/'.join(k):sum(1 for r in selected if (r.get('language','en'),r.get('task_domain','unknown'),r.get('question_type','open'),r.get('known_bucket','known'))==k) for k in cells if any((r.get('language','en'),r.get('task_domain','unknown'),r.get('question_type','open'),r.get('known_bucket','known'))==k for r in selected)}},ensure_ascii=False))
if __name__=='__main__': main()
