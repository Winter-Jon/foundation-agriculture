"""Resumable test-only full-tool evaluation against a local model service."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import requests
import yaml

from agrinet.data.full_tool_sft import read_rows, sha, write_json
from agrinet.rag.dual_teacher import diagnostic_evidence
from agrinet.rag.e344_runtime import Runtime
from agrinet.vlm.full_tool_inference import evaluate, FINALIZE


def generation_payload(model, messages, max_tokens):
    # Stop decoding at the first completed call, retaining the actual closing
    # delimiter. The next generation receives the real tool result. Do not
    # extract a first call from an already generated multi-call completion.
    payload = {'model': model, 'messages': messages, 'temperature': 0, 'top_p': 1,
            'max_tokens': max_tokens, 'chat_template_kwargs': {'enable_thinking': False},
            'stop': ['</tool_call>'], 'no_stop_trim': True}
    if any(FINALIZE in str(m.get('content', '')) for m in messages if m['role'] == 'user'):
        payload.pop('stop')
        payload['regex'] = r'<think>[^<>]+</think>\s*<answer>[^<>]+</answer>'
    return payload


def recovery_rows(source: Path) -> dict:
    """Resolve a recovery lineage without regenerating already journaled rows."""
    merged = source / 'merged_predictions.jsonl'
    if merged.exists():
        return {row['id']: row for row in read_rows(merged)}
    current = {row['id']: row for row in read_rows(source / 'journal.jsonl')}
    lineage = source / 'recovery-lineage.json'
    if not lineage.exists():
        return current
    parent = Path(json.loads(lineage.read_text())['parent_evaluation'])
    return {**recovery_rows(parent), **current}


def finalize_recovery(config, output):
    """Materialize an already-completed recovery journal without inference."""
    output = Path(output)
    source_value = config['parameters'].get('recovery_source')
    if not source_value:
        raise ValueError('finalization requires parameters.recovery_source')
    rows = read_rows(config['inputs']['test_manifest'])
    parent_rows = recovery_rows(Path(source_value))
    local_rows = {row['id']: row for row in read_rows(output / 'journal.jsonl')}
    merged = {**parent_rows, **local_rows}
    if {row['id'] for row in rows} != set(merged):
        raise ValueError('recovery lineage does not cover the fixed test manifest')
    (output / 'merged_predictions.jsonl').write_text(
        ''.join(json.dumps(merged[row['id']], ensure_ascii=False) + '\n' for row in rows)
    )
    return merged


def run(config, api_base, model, output):
    inputs, params = config['inputs'], config['parameters']
    rows = read_rows(inputs['test_manifest'])
    cards = {r['id']: r for r in read_rows(inputs['classifier_cards'])} if 'classifier_cards' in inputs else None
    predictions = {} if cards else {r['image_id']: r for r in read_rows(inputs['classifier_predictions'])}
    labels = {} if cards else {r['canonical_class_code']: r['english_name'] for r in json.loads(Path(inputs['label_map']).read_text())}
    if len(rows) != params.get('expected_rows', 1019) or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('evaluation manifest row count or uniqueness failure')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    keys = ('test_manifest', 'classifier_cards') if cards else ('test_manifest', 'classifier_predictions', 'label_map')
    binding = {'model': model, 'inputs': {k: sha(inputs[k]) for k in keys},
               'parameters': params, 'implementation': {str(p): sha(p) for p in [Path(__file__), Path('src/agrinet/vlm/full_tool_inference.py'), Path('src/agrinet/vlm/full_tool.py')]}}
    marker = output / 'binding.json'
    if marker.exists() and json.loads(marker.read_text()) != binding:
        raise ValueError('evaluation resume binding changed')
    write_json(marker, binding)
    recovery_source = params.get('recovery_source')
    source_rows = {}
    retry_ids = None
    if recovery_source:
        parent = Path(recovery_source)
        parent_binding = json.loads((parent / 'binding.json').read_text())
        stable = ('rag_endpoint', 'reference_image_root', 'expected_rows', 'max_new_tokens', 'context_length', 'test_only')
        if parent_binding['model'] != model or parent_binding['inputs'] != binding['inputs']:
            raise ValueError('recovery parent model or input binding changed')
        if any(parent_binding['parameters'].get(key) != params.get(key) for key in stable):
            raise ValueError('recovery inference contract changed')
        source_rows = recovery_rows(parent)
        retry_ids = {key for key, value in source_rows.items() if str(value.get('error', '')).startswith('runtime:')}
        if not retry_ids:
            raise ValueError('recovery parent has no runtime rows')
        write_json(output / 'recovery-lineage.json', {
            'parent_evaluation': str(parent), 'parent_binding_sha256': sha(parent / 'binding.json'),
            'parent_journal_sha256': sha(parent / 'journal.jsonl'), 'retry_ids': sorted(retry_ids),
            'original_runtime_errors': {key: source_rows[key]['error'] for key in sorted(retry_ids)},
        })
    journal = output / 'journal.jsonl'
    done = {r['id']: r for r in read_rows(journal)} if journal.exists() else {}
    runtime = Runtime({}, rag_endpoint=params['rag_endpoint'])

    def one(public):
        row = dict(public)
        row['question'] = 'Identify the agricultural disease or pest shown in this image.'
        prediction = cards[row['id']] if cards else predictions[row['id']]
        if prediction['image_sha256'] != row['image_sha256'] or sha(row['image_path']) != row['image_sha256']:
            raise ValueError('classifier/image binding mismatch')
        card = prediction['response'] if cards else {'top3': [{'code': v['canonical_class_code'], 'name': labels[v['canonical_class_code']],
                          'score': v['confidence']} for v in prediction['topk'][:3]]}
        usage = []
        def generate(messages):
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(api_base.rstrip('/') + '/chat/completions',
                                        json=generation_payload(model, messages, params['max_new_tokens']),
                                        timeout=params.get('request_timeout_seconds', 300))
                if not response.ok:
                    raise RuntimeError('model_http_%s:%s' % (response.status_code, response.text[:1000]))
                payload = response.json()
            usage.append(payload.get('usage', {}))
            return payload['choices'][0]['message'].get('content') or ''
        started = time.monotonic()
        try:
            result = evaluate(row, generate=generate, classifier=lambda: card,
                              retrieve=lambda args: diagnostic_evidence(runtime.retrieve(row, args)),
                              image_root=params['reference_image_root'])
        except Exception as exc:
            result = {'prediction': '', 'protocol_error': None, 'error': f'runtime:{type(exc).__name__}:{exc}'}
        if result.get('protocol_error'):
            result['error'] = 'model_protocol:' + result['protocol_error']
        return {**public, **result, 'route': 'full_tool', 'model_identifier': model,
                'tool_turns': len(result.get('tool_history', [])),
                'latency_seconds': time.monotonic() - started, 'usage': usage}

    with concurrent.futures.ThreadPoolExecutor(max_workers=params['max_concurrent']) as executor:
        pending = [executor.submit(one, row) for row in rows
                   if row['id'] not in done and (retry_ids is None or row['id'] in retry_ids)]
        with journal.open('a') as stream:
            for future in concurrent.futures.as_completed(pending):
                value = future.result()
                stream.write(json.dumps(value, ensure_ascii=False) + '\n')
                stream.flush()
                done[value['id']] = value
                print(json.dumps({'completed': len(done), 'total': len(retry_ids) if retry_ids is not None else len(rows)}), flush=True)
    selected = [row for row in rows if retry_ids is None or row['id'] in retry_ids]
    (output / 'predictions.jsonl').write_text(''.join(json.dumps(done[row['id']], ensure_ascii=False) + '\n' for row in selected))
    if recovery_source:
        merged = {**source_rows, **done}
        (output / 'merged_predictions.jsonl').write_text(''.join(json.dumps(merged[row['id']], ensure_ascii=False) + '\n' for row in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--api-base')
    parser.add_argument('--model')
    parser.add_argument('--output', required=True)
    parser.add_argument('--finalize-recovery', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.finalize_recovery:
        finalize_recovery(config, args.output)
    elif args.api_base and args.model:
        run(config, args.api_base, args.model, args.output)
    else:
        parser.error('--api-base and --model are required unless --finalize-recovery is used')
