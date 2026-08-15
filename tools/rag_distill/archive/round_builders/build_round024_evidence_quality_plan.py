"""Select a tiny, high-margin, route-aware round024 Pilot from reviewed preflight pools."""
import json
from pathlib import Path

reports = [
    Path('outputs/experiments/rag_sft_iteration/candidates/round-015-preflight-balanced4/report.jsonl'),
    Path('outputs/experiments/rag_sft_iteration/candidates/round-011-preflight-unused-zh-pest/preflight_report.jsonl'),
    Path('outputs/experiments/rag_sft_iteration/candidates/round-013-preflight-fresh-en-pest/report.jsonl'),
]
wanted = {
    'rag_open-zh-disease_N04053_P00002': 'balanced_stop',
    'rag_open-zh-pest_N05070_P00001': 'visual_then_balanced',
    'rag_open-en-pest_N05070_P00002': 'visual_then_balanced',
}
rows = {}
for report in reports:
    for line in report.open(encoding='utf-8'):
        row = json.loads(line)
        if row.get('target_id') in wanted:
            row = dict(row)
            row['strategy_id'] = wanted[row['target_id']]
            row['preferred_sequence'] = {'balanced_stop': ['balanced'], 'visual_then_balanced': ['visual', 'balanced']}[row['strategy_id']]
            row['round'] = 24
            row['focus'] = ['evidence_quality', 'rank1_margin', 'route_composition', 'language_isolation']
            rows[row['target_id']] = row
assert set(rows) == set(wanted), (set(wanted) - set(rows))
for row in rows.values():
    check = row['strategy_preflight'][row['strategy_id']]
    assert check['eligible'] and check['rank'] == 1 and check['score'] >= 0.70, row['target_id']
    assert Path(row['query_image']).is_file(), row['query_image']
out_dir = Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan')
out_dir.mkdir(parents=True, exist_ok=True)
source = out_dir / 'round024_evidence_quality_source.jsonl'
plan = out_dir / 'round024_evidence_quality_plan.jsonl'
with source.open('w', encoding='utf-8') as s, plan.open('w', encoding='utf-8') as p:
    for row in rows.values():
        line = json.dumps(row, ensure_ascii=False) + '\n'
        s.write(line); p.write(line)
print(json.dumps({'targets': len(rows), 'strategies': {r['target_id']: r['strategy_id'] for r in rows.values()}, 'scores': {r['target_id']: r['preflight_target_score'] for r in rows.values()}}, ensure_ascii=False))
