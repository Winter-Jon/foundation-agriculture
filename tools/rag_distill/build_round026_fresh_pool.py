"""Build a deterministic fresh bilingual pool for round026 preflight."""
import json
from pathlib import Path

classes = {json.loads(line)['code']: json.loads(line) for line in Path('outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl').open()}
codes = ['N04007', 'N04029', 'N04063', 'N04071', 'N05014', 'N05018', 'N05029', 'N05070']
used = set()
for path in Path('outputs/artifacts/datasets').glob('agrinet-rag-sft-round*-singleimg/data.jsonl'):
    for line in path.open():
        used.add(str(json.loads(line).get('images', [''])[0]))
for path in Path('outputs/experiments/rag_sft_iteration/candidates').glob('**/train/*.jsonl'):
    for line in path.open():
        try:
            row = json.loads(line); used.add(str((row.get('images') or [''])[0]))
        except Exception:
            pass
rows = []
for code in codes:
    item = classes[code]; domain = item['task_domain']
    fresh = sorted(str(p) for p in Path('datasets/AgriNet-1K/all', code).glob(f'{code}_*.jpg') if str(p) not in used)[:6]
    for index, image in enumerate(fresh):
        language = 'en' if index % 2 == 0 else 'zh'
        part = Path(image).stem.split('_')[-1]
        rows.append({'target_id': f'rag_open-{language}-{domain}_{code}_{part}', 'source_sample_id': f'{domain}_{code}_{part}', 'query_image': image, 'class_code': code, 'class_name': item['english_name'], 'class_name_zh': item['chinese_name'], 'task_domain': domain, 'language': language, 'question_type': 'open', 'trajectory_mode': 'standard', 'train_eligible': True, 'max_tool_turns': 3, 'reserve': False, 'generation_route': 'blind_evidence', 'label_visible_to_teacher': False, 'strategy_id': 'visual_then_balanced', 'preferred_sequence': ['visual', 'balanced'], 'retrieval_top_k': 5, 'round': 26, 'focus': ['fresh_unused_image', 'high_margin_preflight', 'bilingual_domain_balance'], 'sample_id': f'{domain}_{code}_{part}'})
out = Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round026_fresh_pool.jsonl'); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
print(json.dumps({'rows': len(rows), 'en': sum(r['language']=='en' for r in rows), 'zh': sum(r['language']=='zh' for r in rows), 'disease': sum(r['task_domain']=='disease' for r in rows), 'pest': sum(r['task_domain']=='pest' for r in rows)}))
