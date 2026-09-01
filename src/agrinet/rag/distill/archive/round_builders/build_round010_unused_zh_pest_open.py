#!/usr/bin/env python3
import json
from pathlib import Path

root = Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan')
classes = {}
for line in Path('outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl').open(encoding='utf-8'):
    item = json.loads(line)
    classes[item['code']] = item
code = 'N05029'
item = classes[code]
image = sorted(Path('datasets/AgriNet-1K/all', code).glob('*.jpg'))[0]
source_id = f'pest_{code}_{image.stem}'
base = {
    'target_id': f'rag_open-zh-pest_{code}_{code}_P00817',
    'source_sample_id': source_id,
    'query_image': str(image),
    'image_sha256': '',
    'class_code': code,
    'class_name': item['english_name'],
    'class_name_zh': item['chinese_name'],
    'task_domain': 'pest', 'language': 'zh', 'question_type': 'open',
    'trajectory_mode': 'standard', 'train_eligible': True, 'max_tool_turns': 3,
    'reserve': False, 'generation_route': 'blind_evidence', 'label_visible_to_teacher': False,
    'strategy_id': 'visual_then_balanced', 'preferred_sequence': ['visual', 'balanced'],
    'retrieval_top_k': 5, 'round': 10,
    'focus': ['evidence_grounded_open_answer', 'unused_target', 'language_isolation'],
    'sample_id': source_id,
}
source_path = root / 'round010_unused_zh_pest_open_source.jsonl'
plan = dict(base)
plan["source_sample_id"] = source_id
plan["sample_id"] = base["target_id"] + "-round010"
plan["candidate_index"] = 1
source_path.write_text(json.dumps(base, ensure_ascii=False) + '\n', encoding='utf-8')
(root / 'round010_unused_zh_pest_open_plan.jsonl').write_text(json.dumps(plan, ensure_ascii=False) + '\n', encoding='utf-8')
print(source_path)
