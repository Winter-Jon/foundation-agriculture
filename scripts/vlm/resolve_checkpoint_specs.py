#!/usr/bin/env python3
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
requested = sys.argv[2].split()
max_steps = int(state.get('max_steps') or 0)
explicit_steps = []
epoch_specs = []
for item in requested:
    if item.startswith(('step:', 'step=')):
        step = int(item.split(item[4], 1)[1])
        if step <= 0 or step > max_steps:
            raise SystemExit(f'checkpoint step out of range: {step} (max_steps={max_steps})')
        explicit_steps.append((item, step))
    else:
        epoch_specs.append(item)
epochs = [int(item.split(':', 1)[0]) for item in epoch_specs]
max_epoch = max(epochs) if epochs else 0
if max_steps <= 0 or (not explicit_steps and max_epoch <= 0):
    raise SystemExit('cannot resolve checkpoint specs from trainer_state.json')
resolved = []
for item, step in explicit_steps:
    resolved.append(f'0:{step}')
for item in epoch_specs:
    epoch = int(item.split(':', 1)[0])
    step = max_steps * epoch
    if step % max_epoch:
        raise SystemExit(f'non-integral checkpoint step for epoch {epoch}')
    resolved.append(f'{epoch}:{step // max_epoch}')
print(' '.join(resolved))
