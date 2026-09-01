#!/usr/bin/env zsh
# Wait for the already-running large-008 collector, then make one immutable
# preflight candidate view.  This script deliberately never launches sampling.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
RUN_ID=large-20260819-large-008
PLAN="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/${RUN_ID}/supplement"
COLLECTION="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection"
DEST="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-20260819-freeze-preflight-v2"
LOG_DIR="outputs/runs/rag/hermes-v2-large008-postflight/logs"
mkdir -p "$LOG_DIR"

# The collector tmux session deliberately keeps an interactive shell after its
# script exits, so session existence is not a completion signal.  Watch the
# actual large-008 collector processes instead.
# Bracket the first literal so pgrep does not match its own command line.
while pgrep -f "[t]ools.rag_distill.collect_hermes_1to1_v2.*${RUN_ID}/supplement" >/dev/null; do
  sleep 60
done

# A vanished tmux pane alone is not success: every manifest must have closed
# accounting before the selector observes the candidate pool.
.venv/bin/python - "$PLAN" "$COLLECTION" "$RUN_ID" <<'PY'
import sys
from pathlib import Path
plan, collection, run_id = map(Path, sys.argv[1:])
missing = []
for target in sorted(plan.glob('*.jsonl')):
    route, label = target.stem.split('-', 1)
    output = collection / f'{run_id}-supplement-{route}-{label}'
    expected = sum(1 for _ in target.open(encoding='utf-8'))
    actual = 0
    for name in ('accepted.jsonl', 'rejected.jsonl'):
        path = output / name
        if path.exists():
            actual += sum(1 for _ in path.open(encoding='utf-8'))
    if actual != expected:
        missing.append({'target': target.name, 'expected': expected, 'accounted': actual})
if missing:
    raise SystemExit(f'collection accounting incomplete: {missing}')
print('all large-008 targets have closed accounting')
PY

[[ ! -e "$DEST" ]] || { print -u2 "preflight destination already exists: $DEST"; exit 1; }
.venv/bin/python -m agrinet.rag.distill.select_hermes_1to1_large_freeze \
  --collection-root "$COLLECTION" \
  --forbidden-hashes outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/isolation/forbidden_image_sha256.json \
  --destination "$DEST"
