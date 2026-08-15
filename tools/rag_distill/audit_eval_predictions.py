#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--file',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    rows=[json.loads(x) for x in args.file.open(encoding='utf-8') if x.strip()]
    out=[]
    for r in rows:
        pred=str(r.get('prediction') or '')
        clean=re.sub(r'<think>.*?</think>',' ',pred,flags=re.S|re.I)
        m=re.search(r'<answer>\s*(.*?)\s*</answer>',clean,flags=re.S|re.I)
        answer=m.group(1).strip() if m else clean.strip()
        out.append({'id':r.get('id'),'language':r.get('language'),'domain':r.get('task_domain'),'question_type':r.get('question_type'),'label_name':r.get('label_name'),'label_aliases':r.get('label_aliases'),'answer':answer,'prediction_tail':pred[-500:]})
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rows':len(out),'out':str(args.out)},ensure_ascii=False))
if __name__=='__main__': main()
