"""Offline class-macro and paired test-only reporting; never used for selection."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from agrinet.data.full_tool_sft import read_rows, write_json


def summarize(predictions, scored, truth):
    by_class = defaultdict(list)
    for r in scored:
        by_class[truth[r['id']]['canonical_class_code']].append(int(r['correct']))
    latency = [r['latency_seconds'] for r in predictions]
    return {'rows': len(scored), 'accuracy': float(np.mean([r['correct'] for r in scored])),
            'class_macro_accuracy': float(np.mean([np.mean(v) for v in by_class.values()])),
            'classes': len(by_class),
            'protocol_errors': sum(bool(r.get('protocol_error')) for r in predictions),
            'runtime_errors': sum(str(r.get('error', '')).startswith('runtime:') for r in predictions),
            'refusals': sum(bool(r.get('insufficient_evidence')) for r in predictions),
            'rag_calls': dict(Counter(r.get('rag_calls', 0) for r in predictions)),
            'retrieval_modes': dict(Counter(m for r in predictions for m in r.get('retrieval_modes', []))),
            'latency_mean_seconds': float(np.mean(latency)), 'latency_p95_seconds': float(np.quantile(latency, .95)),
            'usage': {key: sum(u.get(key, 0) for r in predictions for u in r.get('usage', []))
                      for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}}


def report(directories, truth_path, output):
    truth = {r['id']: r for r in read_rows(truth_path)}
    results, scores = {}, {}
    for label, directory in directories.items():
        directory = Path(directory)
        # Recovery evaluations preserve a local retry journal and materialize
        # the protocol-bound full population separately.
        prediction_path = directory / 'merged_predictions.jsonl'
        if not prediction_path.exists():
            prediction_path = directory / 'predictions.jsonl'
        predictions = read_rows(prediction_path)
        scored = read_rows(directory / 'scored.jsonl')
        if {r['id'] for r in scored} != set(truth) or len(scored) != len(truth):
            raise ValueError('incomplete test report')
        results[label] = summarize(predictions, scored, truth)
        scores[label] = {r['id']: int(r['correct']) for r in scored}
    ids = sorted(truth)
    rng = np.random.default_rng(42)
    comparisons = {}
    for left, right in [('epoch3', 'base'), ('epoch6', 'base'), ('epoch6', 'epoch3')]:
        difference = np.asarray([scores[left][i] - scores[right][i] for i in ids])
        bootstrap = np.asarray([rng.choice(difference, len(ids), replace=True).mean() for _ in range(10000)])
        comparisons[f'{left}-{right}'] = {'accuracy_difference': float(difference.mean()),
            'paired_sample_bootstrap_95ci': np.quantile(bootstrap, [.025, .975]).tolist()}
    write_json(output, {'models': results, 'comparisons': comparisons, 'selection_split': None,
        'predetermined_final': 'epoch6', 'limitation': 'Test-only descriptive reporting on a previously used test set; no checkpoint selection. Sample bootstrap is not a class-cluster confidence interval.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('base', 'epoch3', 'epoch6', 'truth', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    report({k: getattr(a, k) for k in ('base', 'epoch3', 'epoch6')}, a.truth, a.output)
