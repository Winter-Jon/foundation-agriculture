"""Build the bounded two-stage replay-safe curriculum for round023."""
import json
from pathlib import Path

ROOT = Path('outputs/artifacts/datasets')
OUT = ROOT / 'agrinet-rag-sft-round023-curriculum-singleimg'
OUT.mkdir(parents=True, exist_ok=True)
sources = {
    'anchor': ROOT / 'agrinet-rag-sft-round007-merged5-singleimg/data.jsonl',
    'zh_pest': ROOT / 'agrinet-rag-sft-round016-zh-pest-singleimg/data.jsonl',
    'en_pest': ROOT / 'agrinet-rag-sft-round018-en-pest-singleimg/data.jsonl',
}
def read(path):
    return [json.loads(line) for line in path.open()]
anchor = read(sources['anchor'])
additions = read(sources['zh_pest']) + read(sources['en_pest'])
rows = anchor + additions
assert len(anchor) == 5 and len(additions) == 2
assert len({r['sample_id'] for r in rows}) == 7
for stage, stage_rows in [('stage1_anchor', anchor), ('stage2_additions', additions), ('combined', rows)]:
    path = OUT / f'{stage}.jsonl'
    with path.open('w') as f:
        for row in stage_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
manifest = {
    'schema_version': 'agrinet.rag-sft-curriculum/v1',
    'round': 'round023',
    'stages': [
        {'name': 'stage1_anchor', 'rows': 5, 'dataset': str(OUT / 'stage1_anchor.jsonl'), 'purpose': 'replay stable round007 anchor'},
        {'name': 'stage2_additions', 'rows': 2, 'dataset': str(OUT / 'stage2_additions.jsonl'), 'purpose': 'adapt to reviewed Pest additions'},
    ],
    'combined_rows': 7,
    'source_datasets': {k: str(v) for k, v in sources.items()},
    'protocol_unchanged': True,
    'formal_eval_authorized': False,
}
(OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
