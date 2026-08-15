import argparse,json
from collections import defaultdict
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--source',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args()
rows=[json.loads(x) for x in a.source.open(encoding='utf-8') if x.strip()]
groups=defaultdict(list)
for r in rows:
    if r.get('question_type')=='open':
        groups[(r.get('language'),r.get('task_domain'),r.get('known_bucket','known'))].append(r)
selected=[]
for lang in ('en','zh'):
    for dom in ('disease','pest'):
        pool=sorted(groups[(lang,dom,'known')],key=lambda x:str(x.get('id')))
        if len(pool)<2: raise SystemExit(f'{lang}/{dom} insufficient')
        selected.extend(pool[:2])
a.out.parent.mkdir(parents=True,exist_ok=True)
with a.out.open('w',encoding='utf-8') as f:
    for r in selected: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
print(json.dumps({'rows':len(selected),'out':str(a.out)},ensure_ascii=False))
