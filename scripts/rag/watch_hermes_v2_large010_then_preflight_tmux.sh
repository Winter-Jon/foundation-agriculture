#!/usr/bin/env zsh
# Accounting-gated final selector after the active large-010 collection.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
RUN_ID=large-20260819-large-010
PLAN="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/${RUN_ID}/supplement"
COLLECTION="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection"
DEST="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-20260819-freeze-preflight-v4"
LOG_DIR="outputs/runs/rag/hermes-v2-large010-postflight/logs"
mkdir -p "$LOG_DIR"
while pgrep -f "[t]ools.rag_distill.collect_hermes_1to1_v2.*${RUN_ID}/supplement" >/dev/null; do sleep 60; done
.venv/bin/python - "$PLAN" "$COLLECTION" "$RUN_ID" <<'PY'
import sys
from pathlib import Path
plan, collection, run_id = map(Path, sys.argv[1:])
bad=[]
for target in sorted(plan.glob('*.jsonl')):
    route, label=target.stem.split('-', 1)
    out=collection / f'{run_id}-supplement-{route}-{label}'
    expected=sum(1 for _ in target.open(encoding='utf-8'))
    actual=sum(sum(1 for _ in (out/name).open(encoding='utf-8')) for name in ('accepted.jsonl','rejected.jsonl') if (out/name).exists())
    if actual != expected: bad.append((target.name, expected, actual))
if bad: raise SystemExit(f'collection accounting incomplete: {bad}')
print('all large-010 targets have closed accounting')
PY
[[ ! -e "$DEST" ]] || { print -u2 "preflight destination already exists: $DEST"; exit 1; }
.venv/bin/python -m tools.rag_distill.select_hermes_1to1_large_freeze \
  --collection-root "$COLLECTION" \
  --forbidden-hashes outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/isolation/forbidden_image_sha256.json \
  --destination "$DEST"
