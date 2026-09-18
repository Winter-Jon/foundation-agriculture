"""Freeze faithful full-tool rows without rewriting teacher supervision."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import random
import shutil
from pathlib import Path

import yaml


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def verify_faithful(row, trace):
    from agrinet.rag.answer_normalization import resolve
    original = json.loads(Path(trace['original_trajectory_path']).read_text())
    messages = copy.deepcopy(original['messages'])
    final = messages[-1]['content']
    begin, end = final.rindex('<answer>') + 8, final.rindex('</answer>')
    messages[-1]['content'] = final[:begin] + resolve(final[begin:end])['canonical_answer'] + final[end:]
    digests = []
    merged = []
    for message in messages:
        if isinstance(message.get('content'), list):
            text = []
            for part in message['content']:
                if part['type'] == 'text':
                    text.append(part['text'])
                elif part['type'] == 'image_url':
                    raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1], validate=True)
                    digests.append(hashlib.sha256(raw).hexdigest())
                    text.append('<image>')
                else:
                    raise ValueError('unknown multimodal part')
            message['content'] = '\n'.join(text)
        if message['role'] == 'user' and merged and merged[-1]['role'] in {'tool', 'user'}:
            merged[-1]['content'] += '\n' + message['content']
        else:
            merged.append(message)
    if row['messages'] != merged or [sha(p) for p in row['images']] != digests:
        raise ValueError('teacher messages or image bytes changed')


def freeze(config):
    source = Path(config['inputs']['source'])
    output = Path(config['outputs']['artifact'])
    if output.exists():
        raise ValueError(f'refusing existing freeze: {output}')
    rows, lineage = read_rows(source / 'data.jsonl'), read_rows(source / 'lineage.jsonl')
    report = json.loads((source / 'template-check.json').read_text())
    if not report['passed'] or report['data_sha256'] != sha(source / 'data.jsonl'):
        raise ValueError('source template report stale or failed')
    if len(rows) != config['parameters']['expected_rows'] or len(rows) != len(lineage):
        raise ValueError('row count mismatch')
    identities = {key: set() for key in ('image_sha256', 'source_group_id', 'near_duplicate_group_id')}
    isolation = json.loads(Path(config['inputs']['isolation']).read_text())
    excluded = set(isolation['excluded_training_shas'])
    if isolation['unresolved']:
        raise ValueError('unresolved isolation')
    images = {}
    for row, trace in zip(rows, lineage, strict=True):
        verify_faithful(row, trace)
        s = trace['source']
        if not s['sft_eligible'] or s['image_split'] != 'train_candidate' or s['class_role'] != 'known':
            raise ValueError('non-training source')
        if s['image_sha256'] in excluded or sha(s['image_path']) != s['image_sha256']:
            raise ValueError('isolation or query hash failure')
        for key, seen in identities.items():
            if not s.get(key) or s[key] in seen:
                raise ValueError(f'duplicate or absent {key}')
            seen.add(s[key])
        if set(row) != {'messages', 'images', 'tools'}:
            raise ValueError('unexpected student fields')
        original = json.loads(Path(trace['original_trajectory_path']).read_text())
        if not original['training_eligible']:
            raise ValueError('ineligible original')
        for reference in original.get('reference_records', []):
            if sha(reference['path']) != reference['sha256']:
                raise ValueError('reference changed')
        for image in row['images']:
            images[image] = sha(image)
    output.mkdir(parents=True)
    (output / 'images').mkdir()
    copied = set()
    for old, digest in images.items():
        target = output / 'images' / (digest + Path(old).suffix)
        if target not in copied:
            shutil.copyfile(old, target)
            if sha(target) != digest:
                raise ValueError('copy hash mismatch')
            copied.add(target)
    order = list(range(len(rows)))
    random.Random(42).shuffle(order)
    converted, traces = [], []
    for i in order:
        row = copy.deepcopy(rows[i])
        row['images'] = [str(output / 'images' / (images[p] + Path(p).suffix)) for p in row['images']]
        converted.append(row)
        traces.append({**lineage[i], 'conversion_source_row': i})
    for name, values in [('data', converted), ('lineage', traces)]:
        (output / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values))
    lengths = {r['row_index']: r['length'] for r in report['rows']}
    selected = {max(order, key=lengths.get), max(order, key=lambda i: len(rows[i]['images']))}
    for arm in ('known', 'simulated_unknown'):
        for kind in ('open', 'option'):
            selected.add(next(i for i in order if lineage[i]['source']['arm'] == arm and lineage[i]['source']['question_type'] == kind))
    for count in (1, 2, 3):
        selected.add(next(i for i in order if sum(c['function']['name'] == 'agrinet_rag_search' for m in rows[i]['messages'] for c in m.get('tool_calls', [])) == count))
    for i in order:
        if len(selected) >= 16:
            break
        selected.add(i)
    smoke = [r for i, r in zip(order, converted) if i in selected]
    (output / 'smoke.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in smoke))
    for name in ('summary.json', 'deficits.json', 'exclusions.json'):
        shutil.copyfile(source / name, output / name)
    write_json(output / 'manifest.json', {
        'schema_version': 'agrinet.full-tool-sft/v1', 'rows': len(rows), 'seed': 42,
        'source': str(source), 'source_data_sha256': sha(source / 'data.jsonl'),
        'data_sha256': sha(output / 'data.jsonl'), 'lineage_sha256': sha(output / 'lineage.jsonl'),
        'images': {str(p): sha(p) for p in sorted(copied)},
        'isolation_sha256': sha(config['inputs']['isolation']),
        'training_authorized': False, 'smoke_source_rows': sorted(selected),
    })
    print(json.dumps({'rows': len(rows), 'smoke': len(smoke), 'artifact': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    freeze(yaml.safe_load(Path(parser.parse_args().config).read_text()))
