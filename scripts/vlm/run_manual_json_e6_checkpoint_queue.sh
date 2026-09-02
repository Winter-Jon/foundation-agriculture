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
# A recovery run may explicitly reuse the durable artifact root of a failed
# queue.  Its managed process logs remain under the new run directory, while
# smoke/formal routes resume from the original immutable predictions.
QUEUE_ROOT="${QUEUE_ARTIFACT_ROOT:-outputs/runs/vlm/$EXPERIMENT_ID/$QUEUE_RUN_ID/artifacts}"
RAG_API="${RAG_API:-http://127.0.0.1:8078}"
RAG_SERVICE_EXPERIMENT="${RAG_SERVICE_EXPERIMENT:-rag-serve-siglip2-milvus-public-evidence-v2}"
SMOKE_LIMIT="${SMOKE_LIMIT:-64}"
FULL_EVAL_WITHOUT_SMOKE="${FULL_EVAL_WITHOUT_SMOKE:-0}"
CHECKPOINT_SPECS="${CHECKPOINT_SPECS:-2:36 4:72 6:108}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-70000}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-60}"
TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-8}"
# ``local_launch.master_port`` is exported by the registered CLI as
# MASTER_PORT.  Retain TRAIN_MASTER_PORT as an explicit shell override, but
# honour the declarative launch port before falling back to the historical
# default.
TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-${MASTER_PORT:-29680}}"

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

if [[ -n "${PARENT_EXPERIMENT_ID:-}" ]]; then
  TRAIN_DIR=$("$PYTHON_BIN" scripts/vlm/resolve_completed_parent_sft_dir.py "$PARENT_EXPERIMENT_ID" "$MODEL_ROOT")
  echo "resolved completed parent SFT directory: $TRAIN_DIR"
fi

check_training_gpu_capacity() {
  "$PYTHON_BIN" - "$CUDA_VISIBLE_DEVICES" "$MIN_FREE_GPU_MIB" <<'PY'
import shutil, subprocess, sys

visible = [item.strip() for item in sys.argv[1].split(',') if item.strip()]
minimum = int(sys.argv[2])
if len(visible) != 8:
    raise SystemExit(f"HCV DP8 training requires exactly 8 visible GPUs, got {visible}")
if shutil.which("nvidia-smi") is None:
    raise SystemExit("nvidia-smi is required for the pre-training GPU capacity check")
query = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
    check=True, capture_output=True, text=True,
)
free = {}
for line in query.stdout.splitlines():
    parts = [part.strip() for part in line.split(',')]
    if len(parts) == 2:
        free[parts[0]] = int(parts[1])
missing = [gpu for gpu in visible if gpu not in free]
insufficient = {gpu: free[gpu] for gpu in visible if gpu in free and free[gpu] < minimum}
if missing or insufficient:
    raise SystemExit(
        f"insufficient exclusive GPU capacity for DP8 training: missing={missing}, "
        f"free_mib={insufficient}, required_each_mib={minimum}"
    )
print(f"GPU capacity check passed: visible={visible}, minimum_free_mib={minimum}")
PY
}

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

record_failure() {
  local exit_code=$1 line_number=${2:-unknown} command=${3:-unknown}
  # Keep a durable handoff record even if ``set -e`` aborts an external tool.
  # The command string is diagnostic metadata only; it is never re-executed.
  set +e
  write_queue_state "failed:exit=$exit_code:line=$line_number" "${CURRENT_CHECKPOINT:-}"
  printf 'queue failed (exit=%s, line=%s, command=%s)\n' "$exit_code" "$line_number" "$command" >&2
  set -e
}

on_queue_error() {
  local exit_code=$?
  record_failure "$exit_code" "${BASH_LINENO[0]:-unknown}" "${BASH_COMMAND:-unknown}"
  exit "$exit_code"
}
trap on_queue_error ERR

if [[ "${WAIT_FOR_GPU_CAPACITY:-0}" == "1" ]]; then
  while :; do
    write_queue_state waiting_gpu
    if check_training_gpu_capacity; then
      break
    fi
    sleep "$GPU_WAIT_INTERVAL_SECONDS"
  done
else
  check_training_gpu_capacity
fi

RAG_SERVICE_STARTED=0
RAG_SERVICE_RUN_DIR=""
ensure_rag_service() {
  "$PYTHON_BIN" - "$RAG_API" <<'PY'
import json, sys, urllib.request
url = sys.argv[1].rstrip('/') + '/health'
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        payload = json.loads(response.read().decode())
    if payload.get('status') not in {'ok', 'healthy', 'success'}:
        raise SystemExit(f"RAG service health is not ready: {payload}")
except Exception as exc:
    raise SystemExit(f"RAG service is unavailable at {url}: {exc}")
print(f"RAG service health passed: {url}")
PY
}

start_rag_service_if_needed() {
  if ensure_rag_service >/dev/null 2>&1; then
    return
  fi
  local launch_output
  launch_output="$("$REPO_ROOT/.venv/bin/agrinet" rag submit "$RAG_SERVICE_EXPERIMENT" --operation serve --detach)"
  RAG_SERVICE_RUN_DIR="$(printf '%s\n' "$launch_output" | sed -n 's/.*run_dir=//p' | tail -1)"
  [[ -n "$RAG_SERVICE_RUN_DIR" ]] || { echo "could not parse RAG service run directory: $launch_output" >&2; return 1; }
  RAG_SERVICE_STARTED=1
  for _ in $(seq 1 120); do
    if ensure_rag_service >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  echo "RAG service health timeout at $RAG_API" >&2
  return 1
}

stop_started_rag_service() {
  if [[ "$RAG_SERVICE_STARTED" == "1" && -n "$RAG_SERVICE_RUN_DIR" ]]; then
    "$REPO_ROOT/.venv/bin/agrinet" rag stop "$RAG_SERVICE_RUN_DIR" >/dev/null 2>&1 || true
  fi
}
on_queue_exit() {
  local exit_code=$?
  stop_started_rag_service
  return "$exit_code"
}
trap on_queue_exit EXIT

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
  # ``validate_smoke`` is called from an ``if`` condition, where Bash does not
  # reliably propagate ``errexit`` out of a failing simple command.  Return
  # explicitly when strict normalization rejects protocol-error rows; otherwise
  # the next gate would try to read a metrics file that was intentionally never
  # written and mask the real rejection with FileNotFoundError.
  if ! "$PYTHON_BIN" vlm/eval/tools/normalize_answers.py --manifest "$smoke_manifest" --predictions "$output/predictions.jsonl" --output-jsonl "$output/scored.jsonl" --output-metrics "$output/metrics.json" --output-csv "$output/scored.csv" >/dev/null; then
    return 1
  fi
  "$PYTHON_BIN" - "$smoke_manifest" "$output/predictions.jsonl" "$output/metrics.json" "$expected" <<'PY'
import json, sys
from pathlib import Path
manifest = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
rows = [json.loads(line) for line in Path(sys.argv[2]).read_text().splitlines() if line.strip()]
metrics = json.loads(Path(sys.argv[3]).read_text())
protocol = metrics.get('hermes_protocol', {})
expected_ids = [row.get('id') for row in manifest]
actual_ids = [row.get('id') for row in rows]
gates = {'row_count': len(rows) == int(sys.argv[4]) == len(expected_ids), 'unique_ids': len(actual_ids) == len(set(actual_ids)) == int(sys.argv[4]), 'manifest_ids': set(actual_ids) == set(expected_ids), 'error_rows': sum(bool(row.get('error')) for row in rows), 'final_tool_call_rows': sum('<tool_call>' in str(row.get('prediction', '')).lower() for row in rows), 'unparseable_rate': metrics.get('overall_unparseable_rate', 1.0), 'invalid_tool_call_rate': protocol.get('invalid_tool_call_rate', 1.0), 'malformed_tool_call_attempts': int(protocol.get('malformed_tool_call_attempts', -1)), 'terminal_closure_failed': int(protocol.get('terminal_closure_failed', 0))}
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
  write_queue_state resuming_existing_training
else
  write_queue_state training
  [[ "$TRAIN_NPROC_PER_NODE" == "8" ]] || { echo "HCV DP8 training requires TRAIN_NPROC_PER_NODE=8, got $TRAIN_NPROC_PER_NODE" >&2; exit 2; }
  # ms-swift owns its own torchrun invocation from NPROC_PER_NODE/MASTER_PORT.
  # Wrapping ``swift sft`` in an outer torchrun launches eight independent
  # eight-rank jobs, causing competing NCCL stores and connection-refused
  # failures before the first optimizer step.
  NPROC_PER_NODE="$TRAIN_NPROC_PER_NODE" MASTER_PORT="$TRAIN_MASTER_PORT" \
    "$REPO_ROOT/.venv/bin/swift" sft "$TRAINING_CONFIG"
  TRAIN_DIR=$("$PYTHON_BIN" scripts/vlm/find_completed_sft_dir.py "$MODEL_ROOT")
fi
[[ -n "$TRAIN_DIR" && -d "$TRAIN_DIR" ]] || { echo "cannot locate completed training directory" >&2; exit 1; }

# Resolve checkpoint labels against the completed trainer state.
TRAINER_STATE="$TRAIN_DIR/checkpoint-$(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sed 's/^checkpoint-//' | sort -n | tail -1)/trainer_state.json"
[[ -f "$TRAINER_STATE" ]] || { echo "missing trainer_state.json under $TRAIN_DIR" >&2; exit 1; }
CHECKPOINT_SPECS=$("$PYTHON_BIN" scripts/vlm/resolve_checkpoint_specs.py "$TRAINER_STATE" "$CHECKPOINT_SPECS")
echo "Resolved checkpoint specs: $CHECKPOINT_SPECS"
write_queue_state evaluation_handoff "$TRAIN_DIR"
start_rag_service_if_needed

for spec in $CHECKPOINT_SPECS; do
  epoch=${spec%%:*}; step=${spec##*:}; label="epoch-$epoch-checkpoint-$step"; checkpoint="$TRAIN_DIR/checkpoint-$step"
  CURRENT_CHECKPOINT="$label"
  [[ -d "$checkpoint" ]] || { echo "missing required $label at $checkpoint" >&2; exit 1; }
  # Full evaluation is the default experiment mode for an explicitly selected
  # checkpoint.  The optional smoke route remains available only as a cheap
  # protocol diagnostic; it must not silently suppress 618-row evidence.
  if [[ "$FULL_EVAL_WITHOUT_SMOKE" == "1" ]] || run_smoke "$checkpoint" "$label"; then
    write_queue_state formal "$label"
    if ! FORMAL_ROOT="$QUEUE_ROOT/$label/formal-direct" CANDIDATE="$checkpoint" EXPERIMENT_ID="$EXPERIMENT_ID-$label-direct" MANIFEST="$MANIFEST" SGLANG_TP_SIZE=1 SGLANG_DP_SIZE=8 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash scripts/vlm/run_direct_formal618_native_dp8.sh; then
      "$PYTHON_BIN" - "$QUEUE_ROOT/$label/formal_rejected.json" "$label" "direct_formal_failed" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'checkpoint_label': sys.argv[2], 'formal_status': 'rejected', 'reason': sys.argv[3]}, indent=2) + '\n')
PY
      continue
    fi
    if ! FORMAL_ROOT="$QUEUE_ROOT/$label/formal" CANDIDATE="$checkpoint" EXPERIMENT_ID="$EXPERIMENT_ID-$label" MANIFEST="$MANIFEST" MAX_TOOL_TURNS=5 INVALID_TOOL_CALL_POLICY=strict CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" RAG_API="$RAG_API" bash scripts/vlm/run_rag_formal618_native_dp8.sh; then
      "$PYTHON_BIN" - "$QUEUE_ROOT/$label/formal_rejected.json" "$label" "rag_formal_protocol_or_runtime_failed" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'checkpoint_label': sys.argv[2], 'formal_status': 'rejected', 'reason': sys.argv[3]}, indent=2) + '\n')
PY
      continue
    fi
    # Promotion is deliberately separate from execution: formal outputs remain
    # available for diagnosis, but an effect/retention failure cannot be
    # mistaken for an accepted HCV checkpoint.
    # A promotion denial is an expected experiment outcome, not a queue
    # runtime failure. Keep it in an if-condition so the global ERR trap
    # cannot terminate the remaining checkpoint evaluations.
    if ! "$PYTHON_BIN" src/agrinet/research/hcv/promotion_gate.py \
      --direct-review "$QUEUE_ROOT/$label/formal-direct/artifacts/candidate_vs_m1_direct_paired_review.json" \
      --rag-review "$QUEUE_ROOT/$label/formal/artifacts/candidate_vs_raw_base_rag_paired_review.json" \
      --output "$QUEUE_ROOT/$label/promotion_gate.json"; then
      echo "HCV promotion gate rejected $label; formal evidence preserved" >&2
    fi
  else
    "$PYTHON_BIN" - "$QUEUE_ROOT/$label/skipped_formal.json" "$label" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'checkpoint_label': sys.argv[2], 'formal_status': 'skipped', 'reason': 'strict_smoke_gate_failed'}, indent=2) + '\n')
PY
  fi
done
write_queue_state complete
