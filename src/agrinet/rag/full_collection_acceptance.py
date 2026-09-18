"""Select final per-class quotas only after actual template acceptance."""
import argparse
import json
from collections import Counter
from pathlib import Path

from agrinet.rag.e344_prepare import rows, digest, PRIORITY, IDENTITIES
from agrinet.rag.e344_pipeline import write, write_rows

CHECKS = ('final_present', 'final_supervised', 'all_calls_supervised',
          'all_evidence_present', 'evidence_masked', 'all_images_present', 'image_tokens_masked')


def accept(root):
    root = Path(root)
    data = list(rows(root / 'data.jsonl'))
    lineage = list(rows(root / 'lineage.jsonl'))
    template_report = json.loads((root / 'template-check.json').read_text())
    if template_report.get('data_sha256') != digest(root / 'data.jsonl'):
        raise ValueError('template_input_hash_mismatch')
    checks = template_report['rows']
    if len(data) != len(lineage) or len(checks) != len(data):
        raise ValueError('incomplete_template_or_lineage')
    if [r['row_index'] for r in checks] != list(range(len(data))):
        raise ValueError('template_row_order_mismatch')
    sources = list(rows(root.parent / 'pilot-source.jsonl'))
    classes = sorted({r['canonical_class_code'] for r in sources})
    available = Counter()
    selected, selected_lineage, excluded = [], [], []
    used = {k: set() for k in IDENTITIES}
    targets = {(arm, question): quota for arm, question, quota in PRIORITY}
    for row, trace, check in zip(data, lineage, checks, strict=True):
        source = trace['source']
        key = (source['canonical_class_code'], source['arm'], source['question_type'])
        if not all(check.get(k, False) for k in CHECKS):
            excluded.append({'sample_id': source['sample_id'], 'reason': check.get('error', 'template_check_failed')})
            continue
        if any(source[k] in used[k] for k in IDENTITIES):
            raise ValueError('accepted_identity_collision')
        for k in IDENTITIES:
            used[k].add(source[k])
        available[key] += 1
        if available[key] <= targets[key[1:]]:
            selected.append(row)
            selected_lineage.append(trace)
    deficits = [{'canonical_class_code': code, 'arm': arm, 'question_type': question,
                 'target': quota, 'available': available[code, arm, question],
                 'deficit': max(0, quota - available[code, arm, question])}
                for code in classes for arm, question, quota in PRIORITY]
    write_rows(root / 'quota-data.jsonl', selected)
    write_rows(root / 'quota-lineage.jsonl', selected_lineage)
    write(root / 'quota-deficits.json', deficits)
    write(root / 'template-exclusions.json', excluded)
    result = {'classes': len(classes), 'target': sum(r['target'] for r in deficits),
              'accepted_within_quota': len(selected), 'template_accepted': sum(available.values()),
              'deficit': sum(r['deficit'] for r in deficits),
              'input_sha256': {name: digest(root / name) for name in
                               ('data.jsonl', 'lineage.jsonl', 'template-check.json')}}
    write(root / 'quota-summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    print(json.dumps(accept(parser.parse_args().root)))
