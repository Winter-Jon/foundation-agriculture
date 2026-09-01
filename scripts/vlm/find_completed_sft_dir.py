from pathlib import Path
import sys

root = Path(sys.argv[1])
candidates = []
for run in root.glob('v*-*'):
    if not run.is_dir():
        continue
    checkpoints = sorted(
        (item for item in run.glob('checkpoint-*') if (item / 'trainer_state.json').is_file()),
        key=lambda item: int(item.name.split('-', 1)[1]),
    )
    if checkpoints:
        candidates.append((checkpoints[-1].stat().st_mtime, run))
if not candidates:
    raise SystemExit(f'no completed training directory with trainer_state.json under {root}')
print(max(candidates, key=lambda item: item[0])[1])
