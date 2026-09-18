"""Own one local model service and run smoke or formal evaluation."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import requests
import yaml

from agrinet.data.full_tool_sft import read_rows, sha, write_json
from agrinet.vlm.full_tool_evaluation import finalize_recovery, run as evaluate


def run(config, checkpoint, mode):
    root = Path(config['outputs']['run_root']) / os.environ['AGRINET_RUN_ID']
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    artifact = Path(config['inputs']['artifact'])
    if mode == 'smoke':
        smoke = read_rows(artifact / 'smoke.jsonl')
        lineage = read_rows(artifact / 'lineage.jsonl')
        data = read_rows(artifact / 'data.jsonl')
        lookup = {json.dumps(r, sort_keys=True): trace for r, trace in zip(data, lineage)}
        public, cards = [], []
        for row in smoke:
            trace = lookup[json.dumps(row, sort_keys=True)]
            source = trace['source']
            original = json.loads(Path(trace['original_trajectory_path']).read_text())
            response = next(e['response'] for e in original['tool_trace'] if e['call']['function']['name'] == 'agrinet_classifier_predict' and 'protocol_error' not in e['response'])
            public.append({'id': source['sample_id'], 'image_path': source['image_path'],
                           'image_sha256': source['image_sha256'], 'question': 'Identify the agricultural disease or pest shown in this image.', 'language': 'en'})
            cards.append({'id': source['sample_id'], 'image_sha256': source['image_sha256'], 'response': response})
        for name, values in [('public', public), ('cards', cards)]:
            (root / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in values))
        config['inputs']['test_manifest'] = str(root / 'public.jsonl')
        config['inputs']['classifier_cards'] = str(root / 'cards.jsonl')
        config['parameters']['expected_rows'] = 16
    service = root / 'service'
    name = 'full-tool-' + mode
    command = ['.venv/bin/python', 'vlm/eval/tools/sglang_service.py', '--model-path', checkpoint,
               '--served-model-name', name, '--run-dir', str(service), '--tp-size', '1', '--dp-size', '8',
               '--max-running-requests', str(config['parameters']['max_concurrent']),
               '--context-length', str(config['parameters']['context_length'])]
    child = None
    try:
        with (root / 'logs/service.log').open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(root / 'service-owner.json', {'pid': child.pid, 'checkpoint': checkpoint})
            api = None
            for _ in range(450):
                if child.poll() is not None:
                    raise RuntimeError('model service exited during startup')
                status = service / 'service_status.json'
                if status.exists():
                    api = json.loads(status.read_text()).get('api_base')
                    if api:
                        break
                time.sleep(2)
            if not api:
                raise RuntimeError('model service readiness timeout')
            evaluate(config, api, name, root / 'evaluation')
        if mode == 'formal':
            truth = config['inputs'].get('private_truth')
            if not isinstance(truth, str) or not truth:
                raise ValueError('formal evaluation requires inputs.private_truth')
            score = [
                '.venv/bin/python', 'scripts/vlm/score_open_agri_v3_private.py',
                '--predictions', str(root / 'evaluation/merged_predictions.jsonl' if config['parameters'].get('recovery_source') else root / 'evaluation/predictions.jsonl'), '--truth', truth,
                '--output-jsonl', str(root / 'evaluation/scored.jsonl'),
                '--output-metrics', str(root / 'evaluation/metrics.json'),
            ]
            with (root / 'logs/score.log').open('w') as log:
                subprocess.run(score, check=True, stdout=log, stderr=subprocess.STDOUT)
        prediction_path = root / 'evaluation/merged_predictions.jsonl' if config['parameters'].get('recovery_source') else root / 'evaluation/predictions.jsonl'
        rows = read_rows(prediction_path)
        runtime_errors = [r for r in rows if str(r.get('error', '')).startswith('runtime:')]
        if runtime_errors:
            raise RuntimeError(f'{len(runtime_errors)} infrastructure errors; inspect predictions')
        if mode == 'smoke':
            completed = sum(not r.get('protocol_error') for r in rows)
            if completed == 0:
                raise RuntimeError('no complete full-tool smoke trajectory')
            write_json(artifact / 'smoke-acceptance.json', {'passed': True, 'data_sha256': sha(artifact / 'data.jsonl'),
                       'checkpoint': checkpoint, 'evaluation': str(root / 'evaluation'),
                       'rows': len(rows), 'complete_trajectories': completed,
                       'protocol_errors': len(rows) - completed,
                       'max_new_tokens': config['parameters']['max_new_tokens'],
                       'context_length': config['parameters']['context_length']})
        write_json(root / 'evaluation-status.json', {'status': 'complete', 'rows': len(rows), 'mode': mode})
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def finalize(config):
    """Score a completed recovery journal without starting a model service."""
    root = Path(config['outputs']['run_root']) / os.environ['AGRINET_RUN_ID']
    finalize_recovery(config, root / 'evaluation')
    truth = config['inputs'].get('private_truth')
    if not isinstance(truth, str) or not truth:
        raise ValueError('formal recovery finalization requires inputs.private_truth')
    score = [
        '.venv/bin/python', 'scripts/vlm/score_open_agri_v3_private.py',
        '--predictions', str(root / 'evaluation/merged_predictions.jsonl'), '--truth', truth,
        '--output-jsonl', str(root / 'evaluation/scored.jsonl'),
        '--output-metrics', str(root / 'evaluation/metrics.json'),
    ]
    with (root / 'logs/score.log').open('w') as log:
        subprocess.run(score, check=True, stdout=log, stderr=subprocess.STDOUT)
    rows = read_rows(root / 'evaluation/merged_predictions.jsonl')
    runtime_errors = [row for row in rows if str(row.get('error', '')).startswith('runtime:')]
    if runtime_errors:
        raise RuntimeError(f'{len(runtime_errors)} infrastructure errors; inspect predictions')
    write_json(root / 'evaluation-status.json', {'status': 'complete', 'rows': len(rows), 'mode': 'formal_recovery_finalization'})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--mode', choices=['smoke', 'formal'], required=True)
    p.add_argument('--finalize-recovery', action='store_true')
    a = p.parse_args()
    config = yaml.safe_load(Path(a.config).read_text())
    if a.finalize_recovery:
        finalize(config)
    else:
        run(config, a.checkpoint, a.mode)
