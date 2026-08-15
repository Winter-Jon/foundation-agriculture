#!/usr/bin/env python3
import json
from pathlib import Path

src = Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/targeted_open_pest_source.jsonl')
out = Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round009_blind_replacement_source.jsonl')
rows = [json.loads(line) for line in src.open(encoding='utf-8') if line.strip()]
source = next(r for r in rows if r.get('target_id') == 'rag_open-zh-pest_N05023_N05023_P00001')
row = dict(source)
row['generation_route'] = 'blind_evidence'
row['label_visible_to_teacher'] = False
row['strategy_id'] = 'visual_then_balanced'
row['preferred_sequence'] = ['visual', 'balanced']
row['round'] = 9
row['focus'] = ['evidence_grounded_open_answer', 'pest_hard_negative', 'language_isolation']
row['sample_id'] = 'pest_N05023_N05023_P00001'
row.pop('strategy_preflight', None)
plan = dict(row)
plan['source_sample_id'] = row['sample_id']
plan['sample_id'] = 'rag_open-zh-pest_N05023_N05023_P00001-round009-blind'
plan['candidate_index'] = 1
plan_path = out.with_name('round009_blind_replacement_plan.jsonl')
plan_path.write_text(json.dumps(plan, ensure_ascii=False) + '\n', encoding='utf-8')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
print(out)
