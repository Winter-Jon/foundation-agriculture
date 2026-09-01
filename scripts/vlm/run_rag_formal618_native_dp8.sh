#!/usr/bin/env bash
# RAG-only formal 618 evaluation: mixed candidate and raw base on native DP8.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
MANIFEST="${MANIFEST:-outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl}"
CANDIDATE="${CANDIDATE:?CANDIDATE is required}"
EXPERIMENT_ID="${EXPERIMENT_ID:-vlm-rag-b2-m2-direct3-rag1-formal618-dp8-v1}"
RUN_ID="${RUN_ID:-formal618-native-dp8-$(date +%Y%m%d-%H%M%S)}"
ROOT="${FORMAL_ROOT:-outputs/runs/vlm/$EXPERIMENT_ID/$RUN_ID}"
EVAL_ROOT="$ROOT/artifacts"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TP_SIZE="${SGLANG_TP_SIZE:-1}"; DP_SIZE="${SGLANG_DP_SIZE:-8}"
RAG_API="${RAG_API:-http://127.0.0.1:8078}"
# The frozen trajectories have one retrieval, while the served model may make
# useful public follow-ups. Use a five-turn formal budget; evaluator v3
# records all protocol events. Invalid calls receive the existing strict
# runtime handling, while every returned row is included in scoring.
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-5}"
INVALID_TOOL_CALL_POLICY="${INVALID_TOOL_CALL_POLICY:-strict}"
[[ "$INVALID_TOOL_CALL_POLICY" == "strict" ]] || { echo "formal evaluation requires INVALID_TOOL_CALL_POLICY=strict" >&2; exit 2; }
[[ -f "$MANIFEST" && -d "$CANDIDATE" ]] || { echo "missing manifest or checkpoint" >&2; exit 2; }
mkdir -p "$ROOT/logs" "$EVAL_ROOT"

score_route() {
  local output="$EVAL_ROOT/$1"
  # Scoring is performed by this validator.  A completed fresh evaluator has
  # predictions before metrics, so metrics must not be a precondition.
  [[ -f "$output/predictions.jsonl" ]] || return 1
  # One full-coverage metric: protocol-error rows are forced incorrect while
  # counts and rates are retained in metrics.json.
  "$PYTHON_BIN" vlm/eval/tools/normalize_answers.py --scoring-policy final-answer-strict-v2 --manifest "$MANIFEST" --predictions "$output/predictions.jsonl" --output-jsonl "$output/scored.jsonl" --output-metrics "$output/metrics.json" --output-csv "$output/scored.csv" >/dev/null
}

run_route() {
  local route=$1 model=$2 output="$EVAL_ROOT/$1" service="$ROOT/services/$1"
  # A finalized predictions file is immutable evidence.  Validate it, even if
  # the validation rejects a protocol error; never relaunch it and silently
  # retry a failed sample on a formal resume.
  if [[ -f "$output/predictions.jsonl" ]]; then
    score_route "$route"
    echo "resume: diagnostic metrics refreshed $route"
    return
  fi
  mkdir -p "$output" "$service"
  rm -f "$service/service_status.json" "$service/service_exit.json" "$service/service.pid"
  setsid "$PYTHON_BIN" vlm/eval/tools/sglang_service.py --model-path "$model" --served-model-name "agrinet-$route" --run-dir "$service" --tp-size "$TP_SIZE" --dp-size "$DP_SIZE" --max-running-requests 24 >"$service/launcher.log" 2>&1 &
  local manager=$! api_base=""
  for _ in {1..420}; do
    [[ -f "$service/service_status.json" ]] && api_base=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("api_base", ""))' "$service/service_status.json")
    [[ -n "$api_base" ]] && break
    kill -0 "$manager" 2>/dev/null || { tail -n 80 "$service/launcher.log" >&2; return 1; }
    sleep 2
  done
  [[ -n "$api_base" ]] || { kill "$manager" 2>/dev/null || true; echo "service health timeout" >&2; return 1; }
  local -a args=(--manifest "$MANIFEST" --output "$output/predictions.jsonl" --repo-root "$REPO_ROOT" --model "agrinet-$route" --api-base "$api_base" --rag-api "$RAG_API" --top-k 3 --max-tool-turns "$MAX_TOOL_TURNS" --disable-forced-first-call --invalid-tool-call-policy "$INVALID_TOOL_CALL_POLICY" --max-concurrent 24 --request-retries 2 --snapshot-every 1 --max-new-tokens 512)
  [[ -f "$output/predictions.jsonl.run.json" ]] && args+=(--resume)
  set +e; "$PYTHON_BIN" vlm/eval/tools/run_qwen3_vl_rag_sglang_eval.py "${args[@]}"; local rc=$?; kill -TERM -- "-$manager" 2>/dev/null || true; wait "$manager" 2>/dev/null; set -e
  (( rc == 0 )) || return "$rc"
  score_route "$route"
}

run_route candidate_rag "$CANDIDATE"
run_route raw_base_rag models/Qwen3-VL-4B-Instruct
"$PYTHON_BIN" tools/rag_distill/review_matched_diagnostic.py --candidate "$EVAL_ROOT/candidate_rag/scored.jsonl" --baseline "$EVAL_ROOT/raw_base_rag/scored.jsonl" --out "$EVAL_ROOT/candidate_vs_raw_base_rag_paired_review.json" --bootstrap-samples 10000 --seed 20260819
"$PYTHON_BIN" - "$EVAL_ROOT" "$MANIFEST" "$CANDIDATE" "$TP_SIZE" "$DP_SIZE" "$RAG_API" "$MAX_TOOL_TURNS" "$INVALID_TOOL_CALL_POLICY" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, manifest = Path(sys.argv[1]), Path(sys.argv[2])
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
routes = {}
for name in ('candidate_rag', 'raw_base_rag'):
    route = root / name
    predictions = [json.loads(line) for line in (route / 'predictions.jsonl').open()]
    routes[name] = {
        'predictions_sha256': digest(route / 'predictions.jsonl'),
        'metrics_sha256': digest(route / 'metrics.json'),
        'metrics': json.loads((route / 'metrics.json').read_text()),
        'protocol': {
            'forced_tool_samples': sum(int(row.get('forced_tool_turns', 0)) > 0 for row in predictions),
            'forced_tool_turns': sum(int(row.get('forced_tool_turns', 0)) for row in predictions),
            'invalid_tool_call_samples': sum(bool(row.get('protocol', {}).get('has_invalid_tool_call')) for row in predictions),
            'malformed_tool_call_attempts': sum(int(row.get('protocol', {}).get('malformed_tool_call_attempts', 0)) for row in predictions),
            'noncanonical_recovered_calls': sum(int(row.get('protocol', {}).get('noncanonical_recovered_calls', 0)) for row in predictions),
            'post_budget_tool_attempts': sum(int(row.get('protocol', {}).get('post_budget_tool_attempts', 0)) for row in predictions),
            'terminal_closure_used': sum(int(row.get('protocol', {}).get('terminal_closure_used', 0)) for row in predictions),
            'terminal_closure_failed': sum(int(row.get('protocol', {}).get('terminal_closure_failed', 0)) for row in predictions),
            'answer_format_corrections': sum(int(row.get('protocol', {}).get('answer_format_corrections', 0)) for row in predictions),
            'error_rows': sum(bool(row.get('error')) for row in predictions),
            'final_tool_call_rows': sum('<tool_call>' in str(row.get('prediction', '')) for row in predictions),
        },
    }
summary = {'schema_version': 'agrinet.rag-formal-evaluation-native-sglang-dp8/v4', 'manifest': str(manifest), 'manifest_sha256': digest(manifest), 'rows': 618, 'candidate_checkpoint': sys.argv[3], 'parallelism': {'backend': 'sglang.launch_server', 'tensor_parallel_size': int(sys.argv[4]), 'data_parallel_size': int(sys.argv[5])}, 'concurrency': {'rag': 24}, 'rag_api': sys.argv[6], 'max_tool_turns': int(sys.argv[7]), 'invalid_tool_call_policy': sys.argv[8], 'terminal_policy': 'one-terminal-closure-v1', 'scoring_protocol': {'version': 'full-coverage-with-protocol-errors/v1', 'rule': 'all manifest-aligned rows are scored; protocol-error rows are counted incorrect and reported separately', 'protocol_errors_block_metrics_or_bootstrap': False}, 'request_retries': 2, 'bootstrap': {'samples': 10000, 'seed': 20260819}, 'routes': routes, 'candidate_vs_raw_base': json.loads((root / 'candidate_vs_raw_base_rag_paired_review.json').read_text())}
(root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'rows': 618, 'rag_delta_pp': summary['candidate_vs_raw_base']['paired_delta_pp']}))
PY
