#!/usr/bin/env zsh
# Wait for the existing large-009 targeted collection, then run one strict
# selector preflight.  This script never launches collection itself.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
RUN_ID=large-20260819-large-009
PLAN="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/${RUN_ID}/supplement"
COLLECTION="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection"
DEST="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-20260819-freeze-preflight-v3"
LOG_DIR="outputs/runs/rag/hermes-v2-large009-postflight/logs"
mkdir -p "$LOG_DIR"

# Brackets prevent pgrep from matching the watcher command itself.
while pgrep -f "[t]ools.rag_distill.collect_hermes_1to1_v2.*${RUN_ID}/supplement" >/dev/null; do
  sleep 60
done

.venv/bin/python - "$PLAN" "$COLLECTION" "$RUN_ID" <<'PY'
import sys
from pathlib import Path
plan, collection, run_id = map(Path, sys.argv[1:])
incomplete = []
for target in sorted(plan.glob('*.jsonl')):
    route, label = target.stem.split('-', 1)
    output = collection / f'{run_id}-supplement-{route}-{label}'
    expected = sum(1 for _ in target.open(encoding='utf-8'))
    accounted = 0
    for name in ('accepted.jsonl', 'rejected.jsonl'):
        path = output / name
        if path.exists():
            accounted += sum(1 for _ in path.open(encoding='utf-8'))
    if accounted != expected:
        incomplete.append({'target': target.name, 'expected': expected, 'accounted': accounted})
if incomplete:
    raise SystemExit(f'collection accounting incomplete: {incomplete}')
print('all large-009 targets have closed accounting')
PY

[[ ! -e "$DEST" ]] || { print -u2 "preflight destination already exists: $DEST"; exit 1; }
.venv/bin/python -m tools.rag_distill.select_hermes_1to1_large_freeze \
  --collection-root "$COLLECTION" \
  --forbidden-hashes outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/isolation/forbidden_image_sha256.json \
  --destination "$DEST"
