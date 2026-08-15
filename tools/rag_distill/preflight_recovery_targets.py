#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import request
from agrinet.data.retrieval_strategies import STRATEGIES, assign_attempt_strategy

def read_jsonl(path): return [json.loads(x) for x in Path(path).open() if x.strip()]
def write_jsonl(path, rows):
    with Path(path).open('w') as stream:
        for row in rows: print(json.dumps(row, ensure_ascii=False), file=stream)
def norm(value): return str(value or '').lower().replace('_', ' ').strip()
def search(api, row, retrieval_type, top_k):
    body=json.dumps({'top_k':top_k,'preset':retrieval_type,'text':'visible crop traits','image_path':row['query_image']}).encode()
    req=request.Request(api.rstrip('/')+f'/search/{retrieval_type}',data=body,headers={'Content-Type':'application/json'},method='POST')
    with request.urlopen(req,timeout=180) as response: raw=json.loads(response.read())
    return raw.get('hybrid') or raw.get('text_vector') or raw.get('image_vector') or []

def preflight_row(api, row):
    aliases={norm(row.get('class_name')),norm(row.get('class_name_zh'))}; checks={}; search_cache={}
    for strategy_id,spec in STRATEGIES.items():
        retrieval_type=spec['sequence'][0]; top_k=spec['top_k']; cache_key=(retrieval_type,top_k)
        try:
            if cache_key not in search_cache: search_cache[cache_key]=search(api,row,retrieval_type,top_k)
            hits=search_cache[cache_key]
            match=next((hit for hit in hits if aliases & {norm(hit.get('english_name')),norm(hit.get('chinese_name'))}),None)
            rank=int(match.get('rank') or 99) if match else 99; score=round(float(match.get('distance') or 0),2) if match else 0.0; refs=(match or {}).get('local_reference_images') or []
            checks[strategy_id]={'eligible':bool(match and rank<=top_k and score>=.70 and refs),'rank':rank,'score':score,'top_k':top_k,'retrieval_type':retrieval_type}
        except Exception as exc:
            checks[strategy_id]={'eligible':False,'rank':99,'score':0.0,'top_k':top_k,'retrieval_type':retrieval_type,'error':str(exc)}
    visual=checks['visual_then_balanced']
    return {**row,'strategy_preflight':checks,'preflight_scope':'visual_target_eligibility_with_service_smoke',
            'preflight_eligible':visual['eligible'],'preflight_target_rank':visual['rank'],'preflight_target_score':visual['score']}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--targets',required=True); parser.add_argument('--output',required=True); parser.add_argument('--report',required=True); parser.add_argument('--rag-api',default='http://127.0.0.1:8077'); parser.add_argument('--max-concurrent',type=int,default=4); parser.add_argument('--attempts-per-target',type=int,default=1); parser.add_argument('--per-cell',type=int,default=4); parser.add_argument('--require-full-quota',action='store_true'); args=parser.parse_args()
    if args.attempts_per_target < 1: parser.error('--attempts-per-target must be >= 1')
    targets=read_jsonl(args.targets); audit=[]
    print(f'Preflighting {len(targets)} targets with max_concurrent={args.max_concurrent}',flush=True)
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
        futures={pool.submit(preflight_row,args.rag_api,row):index for index,row in enumerate(targets)}
        completed={}
        for count,future in enumerate(as_completed(futures),1):
            index=futures[future]; completed[index]=future.result()
            print(f'[{count}/{len(targets)}] target_id={completed[index].get("target_id")} eligible={completed[index]["preflight_eligible"]}',flush=True)
    audit=[completed[index] for index in range(len(targets))]
    eligible=[]; seen=set()
    for row in audit:
        key=(str(row.get('sample_id') or row.get('source_sample_id') or row.get('target_id')),str(row.get('query_image') or ''))
        if row['preflight_eligible'] and key not in seen:
            seen.add(key); eligible.append(row)
    cells=defaultdict(list)
    for row in eligible: cells[(row['trajectory_mode'],row['question_type'],row['language'],row['task_domain'])].append(row)
    selected=[]; shortfalls={}
    for cell,rows in sorted(cells.items()):
        rows.sort(key=lambda x:(x['preflight_target_rank'],-x['preflight_target_score'],x['reserve'],x['target_id']))
        if cell[1]=='option' and any(row.get('correct_option') for row in rows): chosen=[next((row for row in rows if row.get('correct_option')==letter),None) for letter in 'ABCD']; chosen=[row for row in chosen if row]
        else: chosen=rows[:args.per_cell]
        selected+=chosen
        if len(chosen)<args.per_cell: shortfalls['/'.join(cell)]=args.per_cell-len(chosen)
    attempts=[]
    for row in selected:
        for i in range(1,args.attempts_per_target+1):
            attempts.append(assign_attempt_strategy(row,i))
    write_jsonl(args.output,attempts); write_jsonl(args.report+'.audit.jsonl',audit)
    report={'planned':len(audit),'eligible_raw':sum(1 for row in audit if row['preflight_eligible']),'eligible_unique':len(eligible),'selected_targets':len(selected),'attempts':len(attempts),'attempts_per_target':args.attempts_per_target,'quota_shortfalls':shortfalls,'full_quota_required':args.require_full_quota}; Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps(report,ensure_ascii=False)); return int(args.require_full_quota and bool(shortfalls))
if __name__=='__main__': raise SystemExit(main())
