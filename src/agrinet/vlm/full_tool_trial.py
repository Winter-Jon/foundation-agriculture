"""Fail-closed managed smoke/train/test-only full-tool trial."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import yaml

from agrinet.data.full_tool_sft import read_rows, sha, write_json, verify_faithful


def checkpoint_at(root, epoch):
    found = []
    for path in Path(root).glob('**/checkpoint-*/trainer_state.json'):
        state = json.loads(path.read_text())
        if abs(float(state['epoch']) - epoch) < 0.001:
            found.append(path.parent)
    if len(found) != 1:
        raise ValueError(f'expected one epoch {epoch} checkpoint, found {len(found)}')
    return found[0]


def run(config, phase):
    root = Path(config['outputs']['run_root']) / os.environ['AGRINET_RUN_ID']
    root.mkdir(parents=True, exist_ok=True)
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    artifact = Path(config['inputs']['artifact'])
    def state(value, **extra):
        write_json(root / 'trial-state.json', {'phase': value, 'updated_at': time.time(), **extra})
    def execute(command, name, env=None):
        with (logs / (name + '.log')).open('a') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                           env={**os.environ, **(env or {})})
    try:
        state('preflight')
        manifest = json.loads((artifact / 'manifest.json').read_text())
        template = json.loads((artifact / 'template-check.json').read_text())
        parity = json.loads((artifact / 'wire-check.json').read_text())
        if not template['passed'] or not parity['passed'] or manifest['data_sha256'] != sha(artifact / 'data.jsonl') or template['data_sha256'] != manifest['data_sha256']:
            raise ValueError('template/freeze/parity gate failed')
        if any(sha(path) != digest for path, digest in manifest['images'].items()):
            raise ValueError('frozen image changed')
        for row, trace in zip(read_rows(artifact / 'data.jsonl'), read_rows(artifact / 'lineage.jsonl'), strict=True):
            verify_faithful(row, trace)
        gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if gpu:
            raise ValueError(f'eight-GPU exclusive gate: active PIDs {gpu}')
        train = yaml.safe_load(Path(config['inputs']['config']).read_text())
        train['output_dir'] = str(root / 'checkpoints')
        if phase == 'smoke':
            train.update(dataset=[str(artifact / 'smoke.jsonl')], max_steps=3, save_strategy='steps', save_steps=3, save_total_limit=1)
        path = root / 'training.yaml'
        path.write_text(yaml.safe_dump(train, sort_keys=False))
        source_files = [Path(config['inputs']['config']), *Path('src/agrinet/vlm').glob('full_tool*.py'),
                        Path('src/agrinet/data/full_tool_sft.py'), Path('scripts/vlm/validate_full_tool_template.py')]
        write_json(root / 'freeze.json', {'data_sha256': manifest['data_sha256'],
                   'training_sha256': sha(path), 'source_files': {str(p): sha(p) for p in source_files},
                   'report_epochs': [3, 6], 'selection_split': None})
        if phase == 'train':
            gate_path = artifact / 'smoke-acceptance.json'
            gate = json.loads(gate_path.read_text())
            if not gate['passed'] or gate['data_sha256'] != manifest['data_sha256']:
                raise ValueError('smoke acceptance missing or stale')
            if any(gate.get(k) != config['parameters'][k] for k in ('max_new_tokens', 'context_length')):
                raise ValueError('smoke generation budget differs from formal protocol')
        state('training', phase_requested=phase)
        execute(['.venv_test/bin/python', '-m', 'swift.cli.main', 'sft', str(path)], 'training',
                {'MAX_PIXELS': '1048576', 'IMAGE_MAX_TOKEN_NUM': '1024'})
        if phase == 'smoke':
            checkpoints = list((root / 'checkpoints').glob('**/checkpoint-3'))
            if len(checkpoints) != 1:
                raise ValueError('three-step checkpoint missing')
            checkpoint = checkpoints[0]
            trainer = json.loads((checkpoint / 'trainer_state.json').read_text())
            import math
            losses = [r['loss'] for r in trainer['log_history'] if 'loss' in r]
            if not losses or not all(math.isfinite(v) for v in losses):
                raise ValueError('smoke loss invalid')
            write_json(root / 'training-smoke.json', {'passed': True, 'checkpoint': str(checkpoint), 'losses': losses,
                       'data_sha256': manifest['data_sha256'], 'inference_pending': True})
            state('training_smoke_complete_inference_pending', checkpoint=str(checkpoint))
        else:
            checkpoints = {str(epoch): str(checkpoint_at(root / 'checkpoints', epoch)) for epoch in (3, 6)}
            write_json(root / 'checkpoints.json', checkpoints)
            state('training_complete_evaluation_pending', checkpoints=checkpoints)
    except Exception as exc:
        state('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--phase', choices=['smoke', 'train'], required=True)
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()), args.phase)
