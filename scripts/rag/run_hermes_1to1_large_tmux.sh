#!/usr/bin/env zsh
# Run one immutable Hermes v2 large plan: two Direct and two Blind RAG workers.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
if [[ ${1:-} == --worker ]]; then
  RUN_ID=${2:?missing run id}; route=${3:?missing route}; file=${4:?missing targets}; cell=${5:?missing cell}
else
  RUN_ID=${1:?usage: run_hermes_1to1_large_tmux.sh <run-id>}
fi
PLAN="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-${RUN_ID}/plan"
COLLECTION="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection"
RUN_DIR="outputs/runs/rag/hermes-large-1to1-${RUN_ID}"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
[[ -f "$PLAN/report.json" ]] || { print -u2 "missing immutable plan: $PLAN"; exit 1; }

if ! /data/home/jiangwentao/.apikeys/bin/apikey env micu_slb >/dev/null 2>&1; then
  print -u2 "micu_slb credential preflight failed"; exit 1
fi
eval "$(/data/home/jiangwentao/.apikeys/bin/apikey env micu_slb)"

worker() {
  local out="$COLLECTION/large-${RUN_ID}-${route}-${cell}"
  if [[ -e "$out/accepted.jsonl" || -e "$out/rejected.jsonl" ]]; then print -u2 "refusing existing output: $out"; return 1; fi
  local args=(.venv/bin/python -m tools.rag_distill.collect_hermes_1to1_v2 --targets "$file" --route "$route" --output-dir "$out" --model gpt-5.6-terra --limit 140 --offset 0)
  [[ "$route" == rag ]] && args+=(--rag-api http://127.0.0.1:8077)
  "${args[@]}" >"$LOG_DIR/${route}-${cell}.log" 2>&1 || { local rc=$?; [[ $rc == 2 ]] || return $rc; }
}
if [[ ${1:-} == --worker ]]; then
  worker
  exit $?
fi

for file in $PLAN/direct-*.jsonl; do
  cell=${file:t}; cell=${cell#direct-}; cell=${cell%.jsonl}
  print -r -- "$file $cell"
done | xargs -n2 -P2 zsh "$0" --worker "$RUN_ID" direct &
direct_pid=$!
for file in $PLAN/rag-*.jsonl; do
  cell=${file:t}; cell=${cell#rag-}; cell=${cell%.jsonl}
  print -r -- "$file $cell"
done | xargs -n2 -P2 zsh "$0" --worker "$RUN_ID" rag &
rag_pid=$!
wait $direct_pid $rag_pid

.venv/bin/python - "$COLLECTION" "$RUN_ID" "$RUN_DIR/large_blind_rejected.jsonl" <<'PY'
import json, sys
from pathlib import Path
from agrinet.data.sft_recovery import write_jsonl
root, run_id, destination = map(Path, sys.argv[1:])
rows=[]
for path in sorted(root.glob(f"large-{run_id}-rag-*/rejected.jsonl")):
    rows.extend(json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
write_jsonl(destination, rows)
print(json.dumps({"rejected":len(rows),"destination":str(destination)}))
PY
ORACLE_PLAN="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-${RUN_ID}/oracle"
if .venv/bin/python -m tools.rag_distill.build_hermes_1to1_oracle_recovery --rejected "$RUN_DIR/large_blind_rejected.jsonl" --collection-root "$COLLECTION" --destination "$ORACLE_PLAN" --cap-to-remaining >"$LOG_DIR/oracle-plan.log" 2>&1; then
  .venv/bin/python -m tools.rag_distill.collect_hermes_1to1_v2 --targets "$ORACLE_PLAN/rag_targets.jsonl" --route rag --oracle --rag-api http://127.0.0.1:8077 --output-dir "$COLLECTION/large-${RUN_ID}-oracle" --model gpt-5.6-terra --limit 9999 --offset 0 >"$LOG_DIR/oracle.log" 2>&1 || { rc=$?; [[ $rc == 2 ]] || exit $rc; }
fi
.venv/bin/python -m tools.rag_distill.select_hermes_1to1_large_freeze --collection-root "$COLLECTION" --forbidden-hashes outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/isolation/forbidden_image_sha256.json --destination "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-${RUN_ID}/freeze-preflight" >"$LOG_DIR/freeze-preflight.log" 2>&1 || true
print "completed run_id=$RUN_ID logs=$LOG_DIR"
