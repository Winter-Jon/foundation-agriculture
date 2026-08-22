#!/usr/bin/env bash
# Train one six-epoch manual-JSON model, then serialize DP8 smoke/formal routes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
TRAINING_CONFIG="${TRAINING_CONFIG:?TRAINING_CONFIG is required}"
EXPERIMENT_ID="${EXPERIMENT_ID:?EXPERIMENT_ID is required}"
QUEUE_RUN_ID="${AGRINET_RUN_ID:?AGRINET_RUN_ID is required}"
MANIFEST="${MANIFEST:-outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl}"
MODEL_ROOT="${MODEL_ROOT:?MODEL_ROOT is required}"
QUEUE_ROOT="outputs/runs/vlm/$EXPERIMENT_ID/$QUEUE_RUN_ID/artifacts"
RAG_API="${RAG_API:-http://127.0.0.1:8077}"
SMOKE_LIMIT="${SMOKE_LIMIT:-64}"
CHECKPOINT_SPECS="${CHECKPOINT_SPECS:-2:36 4:72 6:108}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

[[ -f "$TRAINING_CONFIG" && -f "$MANIFEST" ]] || { echo "missing training config or formal manifest" >&2; exit 2; }
if [[ -n "${HCV_FREEZE_AUDIT:-}" ]]; then
  [[ -f "$HCV_FREEZE_AUDIT" ]] || { echo "missing HCV freeze audit: $HCV_FREEZE_AUDIT" >&2; exit 2; }
  "$PYTHON_BIN" - "$HCV_FREEZE_AUDIT" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("training_authorized") is not True:
    raise SystemExit("HCV freeze audit does not authorize training")
PY
fi
mkdir -p "$QUEUE_ROOT"

if [[ -n "${WAIT_FOR_RUN_DIR:-}" ]]; then
  status_path="$WAIT_FOR_RUN_DIR/status.json"
  while [[ ! -f "$status_path" ]]; do sleep 30; done
  while :; do
    parent_status=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "$status_path")
    case "$parent_status" in
      complete) break ;;
      failed) echo "dependency queue failed: $WAIT_FOR_RUN_DIR" >&2; exit 1 ;;
      pending|running) sleep 60 ;;
      *) echo "unknown dependency status: $parent_status" >&2; exit 1 ;;
    esac
  done
fi

write_queue_state() {
  local phase=$1 checkpoint=${2:-}
  "$PYTHON_BIN" - "$QUEUE_ROOT/queue_state.json" "$phase" "$checkpoint" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
path = Path(sys.argv[1])
prior = json.loads(path.read_text()) if path.exists() else {}
prior.update({'schema_version': 'agrinet.manual-json-checkpoint-queue/v1', 'phase': sys.argv[2], 'checkpoint': sys.argv[3] or None, 'updated_at': datetime.now().astimezone().isoformat()})
path.write_text(json.dumps(prior, ensure_ascii=False, indent=2) + '\n')
PY
}

validate_smoke() {
  local output=$1 expected=$2
  local smoke_manifest="$output/manifest.prefix.jsonl"
  # The evaluator's --limit consumes non-empty manifest rows in source order.
  # Score the smoke predictions against that same bounded manifest, rather than
  # against the 618-row formal manifest.
  "$PYTHON_BIN" - "$MANIFEST" "$smoke_manifest" "$expected" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
limit = int(sys.argv[3])
rows = []
with source.open(encoding='utf-8') as handle:
    for line in handle:
        if line.strip():
            rows.append(line.rstrip('\n'))
            if len(rows) == limit:
                break
if len(rows) != limit:
    raise SystemExit(f'manifest has only {len(rows)} non-empty rows; expected {limit}')
destination.write_text('\n'.join(rows) + '\n', encoding='utf-8')
PY
  "$PYTHON_BIN" vlm/eval/tools/normalize_answers.py --manifest "$smoke_manifest" --predictions "$output/predictions.jsonl" --output-jsonl "$output/scored.jsonl" --output-metrics "$output/metrics.json" --output-csv "$output/scored.csv" >/dev/null
  "$PYTHON_BIN" - "$output/predictions.jsonl" "$output/metrics.json" "$expected" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
metrics = json.loads(Path(sys.argv[2]).read_text())
protocol = metrics.get('hermes_protocol', {})
gates = {'row_count': len(rows) == int(sys.argv[3]), 'unique_ids': len({row.get('id') for row in rows}) == int(sys.argv[3]), 'error_rows': sum(bool(row.get('error')) for row in rows), 'final_tool_call_rows': sum('<tool_call>' in str(row.get('prediction', '')).lower() for row in rows), 'unparseable_rate': metrics.get('overall_unparseable_rate', 1.0), 'terminal_closure_failed': int(protocol.get('terminal_closure_failed', 0))}
failed = {key: value for key, value in gates.items() if value not in (0, True)}
if failed: raise SystemExit(f'smoke protocol gate failed: {failed}')
PY
}

run_smoke() {
  local checkpoint=$1 label=$2 output="$QUEUE_ROOT/$label/smoke" service="$QUEUE_ROOT/$label/services/candidate"
  if [[ -f "$output/metrics.json" ]]; then validate_smoke "$output" "$SMOKE_LIMIT"; return; fi
  mkdir -p "$output" "$service"; write_queue_state smoke "$label"
  setsid "$PYTHON_BIN" vlm/eval/tools/sglang_service.py --model-path "$checkpoint" --served-model-name "agrinet-$label-smoke" --run-dir "$service" --tp-size 1 --dp-size 8 --max-running-requests 24 >"$service/launcher.log" 2>&1 &
  local manager
  local api_base=""
  manager=$!
  for _ in $(seq 1 420); do
    [[ -f "$service/service_status.json" ]] && api_base=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("api_base", ""))' "$service/service_status.json")
    [[ -n "$api_base" ]] && break
    kill -0 "$manager" 2>/dev/null || { tail -80 "$service/launcher.log" >&2; return 1; }
    sleep 2
  done
  [[ -n "$api_base" ]] || { kill "$manager" 2>/dev/null || true; echo "smoke service health timeout" >&2; return 1; }
  set +e
  "$PYTHON_BIN" vlm/eval/tools/run_qwen3_vl_rag_sglang_eval.py --manifest "$MANIFEST" --limit "$SMOKE_LIMIT" --output "$output/predictions.jsonl" --repo-root "$REPO_ROOT" --model "agrinet-$label-smoke" --api-base "$api_base" --rag-api "$RAG_API" --top-k 3 --max-tool-turns 5 --disable-forced-first-call --invalid-tool-call-policy strict --max-concurrent 24 --request-retries 2 --snapshot-every 1 --max-new-tokens 512 --capture-protocol-trace
  local rc=$?
  kill -TERM -- "-$manager" 2>/dev/null || true; wait "$manager" 2>/dev/null; set -e
  (( rc == 0 )) || return "$rc"; validate_smoke "$output" "$SMOKE_LIMIT"
}

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  TRAIN_DIR="${TRAIN_DIR:?TRAIN_DIR is required when SKIP_TRAIN=1}"
else
  write_queue_state training
  "$PYTHON_BIN" -m swift.cli.main sft "$TRAINING_CONFIG"
  TRAIN_DIR=$(find "$MODEL_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'v0-*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
fi
[[ -n "$TRAIN_DIR" && -d "$TRAIN_DIR" ]] || { echo "cannot locate completed training directory" >&2; exit 1; }

for spec in $CHECKPOINT_SPECS; do
  epoch=${spec%%:*}; step=${spec##*:}; label="epoch-$epoch-checkpoint-$step"; checkpoint="$TRAIN_DIR/checkpoint-$step"
  [[ -d "$checkpoint" ]] || { echo "missing required $label at $checkpoint" >&2; exit 1; }
  if run_smoke "$checkpoint" "$label"; then
    write_queue_state formal "$label"
    FORMAL_ROOT="$QUEUE_ROOT/$label/formal" CANDIDATE="$checkpoint" EXPERIMENT_ID="$EXPERIMENT_ID-$label" MANIFEST="$MANIFEST" MAX_TOOL_TURNS=5 INVALID_TOOL_CALL_POLICY=strict CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" RAG_API="$RAG_API" bash scripts/vlm/run_rag_formal618_native_dp8.sh
  else
    "$PYTHON_BIN" - "$QUEUE_ROOT/$label/skipped_formal.json" "$label" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'checkpoint_label': sys.argv[2], 'formal_status': 'skipped', 'reason': 'strict_smoke_gate_failed'}, indent=2) + '\n')
PY
  fi
done
write_queue_state complete
