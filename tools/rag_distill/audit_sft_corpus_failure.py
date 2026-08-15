#!/usr/bin/env python3
"""Audit mixed SFT rows for language contamination and sequence imbalance."""
from __future__ import annotations
import argparse, json, re
from collections import Counter
from pathlib import Path

ZH = re.compile(r"[\u3400-\u9fff]")
EN = re.compile(r"[A-Za-z]")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-file',type=Path,required=True); ap.add_argument('--report',type=Path,required=True); args=ap.parse_args()
    rows=[json.loads(x) for x in args.train_file.open(encoding='utf-8') if x.strip()]
    out=[]; role_counts=Counter(); contaminated=[]; lengths=[]; rag=direct=0
    for i,row in enumerate(rows):
        msgs=row.get('messages') or []; meta=row.get('metadata') or {}
        user=' '.join(str(m.get('content','')) for m in msgs if m.get('role')=='user')
        assistant=' '.join(str(m.get('content','')) for m in msgs if m.get('role')=='assistant')
        lang='zh' if ZH.search(user) else 'en'
        has_tool=any(m.get('role') in ('tool_call','tool_response') for m in msgs)
        rag += int(has_tool); direct += int(not has_tool)
        for m in msgs: role_counts[m.get('role')]+=1
        lengths.append({'sample_id':row.get('sample_id'), 'chars':len(assistant), 'estimated_tokens':len(assistant.split()), 'route':'rag' if has_tool else 'direct', 'language':lang})
        bad=(lang=='en' and ZH.search(assistant)) or (lang=='zh' and not ZH.search(assistant))
        if bad: contaminated.append({'sample_id':row.get('sample_id'),'language':lang,'reason':'assistant_language_mismatch'})
    lengths.sort(key=lambda x:x['chars'],reverse=True)
    report={'rows':len(rows),'rag_rows':rag,'direct_rows':direct,'role_counts':dict(role_counts),'language_contamination_count':len(contaminated),'contaminated_rows':contaminated,'assistant_length_max_chars':lengths[0]['chars'] if lengths else 0,'assistant_length_median_chars':sorted(x['chars'] for x in lengths)[len(lengths)//2] if lengths else 0,'longest_rows':lengths[:20],'hard_valid':not contaminated and len(rows)==80}
    args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'rag':rag,'direct':direct,'contamination':len(contaminated),'max_chars':report['assistant_length_max_chars'],'hard_valid':report['hard_valid']},ensure_ascii=False))
    return 0 if report['hard_valid'] else 1
if __name__=='__main__': raise SystemExit(main())
