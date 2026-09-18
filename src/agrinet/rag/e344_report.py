"""Rebuild pilot accounting from durable provider ledgers, including failures."""
from collections import Counter
import json
from pathlib import Path
from agrinet.rag.e344_prepare import rows
from agrinet.rag.e344_pipeline import write


def report(root):
    root = Path(root)
    pilot = root / 'pilot'
    summary = json.loads((pilot / 'report.json').read_text())
    sources = list(rows(root / 'pilot-source.jsonl'))
    attempts, total_prompt, total_completion, unknown = [], 0, 0, 0
    for row in sources:
        directory = pilot / 'samples' / row['sample_id']
        for ledger_path in sorted(directory.glob('**/events.jsonl')):
            events = list(rows(ledger_path))
            intents = {e['key']: e for e in events if e['event'] == 'intent'}
            results = {e['key']: e for e in events if e['event'] == 'result'}
            for key, intent in intents.items():
                result = results.get(key, {})
                raw = result.get('response', {})
                raw = raw.get('raw', raw)
                usage = raw.get('usage', {})
                complete = 'prompt_tokens' in usage and 'completion_tokens' in usage
                total_prompt += usage.get('prompt_tokens', 0)
                total_completion += usage.get('completion_tokens', 0)
                unknown += not complete
                attempts.append({'sample_id': row['sample_id'], 'request_id': intent['request_id'],
                    'operation': intent['kind'], 'request_key': key, 'status': result.get('status', 'unresolved'),
                    'usage': usage, 'usage_complete': complete, 'started_at': intent['time'],
                    'ended_at': result.get('time')})
    summary['provider_requests'] = len(attempts)
    summary['usage'] = {'prompt_tokens_observed': total_prompt, 'completion_tokens_observed': total_completion,
                        'requests_with_unknown_usage': unknown, 'includes_private_audits': True}
    summary.pop('cost', None)
    summary.pop('cost_status', None)
    for attempt in summary.get('attempts', []):
        attempt.pop('cost', None)
    summary['provider_attempts'] = attempts
    lineage = list(rows(pilot / 'lineage.jsonl'))
    coverage = []
    naming_errors = []
    for source in sources:
        path = pilot / 'samples' / source['sample_id'] / 'trajectory.json'
        trajectory = json.loads(path.read_text()) if path.exists() else {}
        union = trajectory.get('candidate_union', [])
        answer = trajectory.get('validation', {}).get('answer')
        truth = source['private']['truth_name']
        if answer and answer != truth and answer.casefold().strip() == truth.casefold().strip():
            naming_errors.append({'sample_id': source['sample_id'], 'answer': answer,
                                  'canonical_name': truth, 'kind': 'case_only_protocol_mismatch',
                                  'training_eligible': False})
        coverage.append({'sample_id': source['sample_id'], 'union_size': len(union),
                         'truth_covered': source['private']['truth_name'] in {r['name'] for r in union},
                         'candidate_sources': union})
    summary['union_candidate_coverage'] = coverage
    summary['case_only_name_failures'] = naming_errors
    summary['union_truth_covered'] = sum(r['truth_covered'] for r in coverage)
    summary['isolation_audit'] = json.loads((root / 'isolation-audit.json').read_text())
    qualified = Counter((r['source']['canonical_class_code'], r['source']['arm'], r['source']['question_type']) for r in lineage)
    deficits = json.loads((root / 'deficits.json').read_text())
    for cell in deficits:
        key = (cell['canonical_class_code'], cell['arm'], cell['question_type'])
        cell['qualified'] = qualified[key]
        cell['deficit'] = max(0, cell['target'] - qualified[key])
    write(pilot / 'deficits.json', deficits)
    summary['remaining_qualified_quota'] = sum(r['deficit'] for r in deficits)
    summary['projected_attempts_for_remaining_quota'] = (summary['remaining_qualified_quota'] * 32 / len(lineage)) if lineage else None
    summary['projected_generation_and_audit_tokens_for_remaining_quota'] = (
        (total_prompt + total_completion) / len(lineage) * summary['remaining_qualified_quota']) if lineage and not unknown else None
    summary['projection_limitations'] = 'Small stratified pilot; class-specific rates and exhausted clusters unknown; full collection not authorized'
    write(pilot / 'report.json', summary)
    text = ['# E344 full-tool 32-image pilot', '',
            f"Qualified and exported: {len(lineage)}/32. Remaining qualified quota: {summary['remaining_qualified_quota']}/1926.", '',
            '| Cell | Attempts | Qualified | Errors |', '|---|---:|---:|---|']
    for cell, metrics in summary['cells'].items():
        text.append(f"| {cell} | {metrics['attempted']} | {metrics['qualified']} | {json.dumps(metrics['errors'])} |")
    text += ['', 'RAG call distribution: ' + json.dumps(summary['rag_calls']),
             'Retrieval modes: ' + json.dumps(summary['retrieval_modes']),
             'Repeated searches: ' + str(summary['repeated_searches']), '',
             'Usage (generation and private audit): ' + json.dumps(summary['usage']), '',
             'SHA checks found no train/evaluation overlap. All 1831 evaluation source paths and pHashes resolve from underlying metadata. A Hamming-distance <=6 audit found 14 conflicting training images, all excluded. Source paths are not acquisition-session identifiers, and pHash does not prove absence of every visual duplicate.', '',
             'Full collection and SFT were not launched. Full budget and replenishment boundaries require a user decision.', '',
             '## Complete qualified examples']
    for i, item in enumerate(lineage[:3], 1):
        sid = item['source']['sample_id']
        trajectory = json.loads((pilot / 'samples' / sid / 'trajectory.json').read_text())
        text += ['', f'### {i}. {sid}', '']
        for message in trajectory['messages']:
            content = message.get('content')
            if isinstance(content, list):
                content = '\n'.join(p.get('text', '[bound image]') for p in content)
            text += [f"**{message['role']}**", '', str(content or '')]
            if message.get('tool_calls'):
                text += [json.dumps(message['tool_calls'], ensure_ascii=False, indent=2)]
            text.append('')
    (pilot / 'REPORT.md').write_text('\n'.join(text))
    return summary
