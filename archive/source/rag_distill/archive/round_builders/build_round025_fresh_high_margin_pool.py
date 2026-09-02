"""Create a fresh bilingual pool for route-specific Milvus preflight."""
import json
from pathlib import Path

classes = {json.loads(line)['code']: json.loads(line) for line in Path('outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl').open()}
codes = ['N04007', 'N04029', 'N04063', 'N04071', 'N05014', 'N05018', 'N05029', 'N05070']
used = set()
for path in Path('outputs/artifacts/datasets').glob('agrinet-rag-sft-round*-singleimg/data.jsonl'):
    for line in path.open():
        used.add(str(json.loads(line).get('images', [''])[0]))
rows = []
for code in codes:
    item = classes[code]
    domain = item['task_domain']
    images = sorted(Path('datasets/AgriNet-1K/all', code).glob(f'{code}_*.jpg'))
    fresh = [image for image in images if str(image) not in used]
    if not fresh:
        continue
    for language, image in [('en', fresh[0]), ('zh', fresh[1] if len(fresh) > 1 else fresh[0])]:
        rows.append({
            'target_id': f'rag_open-{language}-{domain}_{code}_{image.stem.split("_")[-1]}',
            'source_sample_id': f'{domain}_{code}_{image.stem.split("_")[-1]}',
            'query_image': str(image), 'class_code': code,
            'class_name': item['english_name'], 'class_name_zh': item['chinese_name'],
            'task_domain': domain, 'language': language, 'question_type': 'open',
            'trajectory_mode': 'standard', 'train_eligible': True, 'max_tool_turns': 3,
            'reserve': False, 'generation_route': 'blind_evidence', 'label_visible_to_teacher': False,
            'strategy_id': 'visual_then_balanced', 'preferred_sequence': ['visual', 'balanced'],
            'retrieval_top_k': 5, 'round': 25, 'focus': ['fresh_unused_image', 'high_margin_preflight', 'bilingual_balance'],
            'sample_id': f'{domain}_{code}_{image.stem.split("_")[-1]}',
        })
out = Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round025_fresh_pool.jsonl')
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
print(json.dumps({'rows': len(rows), 'languages': {'en': sum(r['language']=='en' for r in rows), 'zh': sum(r['language']=='zh' for r in rows)}, 'domains': {'disease': sum(r['task_domain']=='disease' for r in rows), 'pest': sum(r['task_domain']=='pest' for r in rows)}}))
