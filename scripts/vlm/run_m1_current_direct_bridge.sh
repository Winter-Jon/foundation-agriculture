#!/usr/bin/env bash
# Evaluate the M1 Direct-only checkpoint under the current public 618 Direct protocol.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
MANIFEST="${MANIFEST:-outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl}"
M1_CHECKPOINT="${M1_CHECKPOINT:-outputs/vlm_sft/qwen3_vl_4b_disease_pest_full_all_e5_len2048_liger_lr1e5_final/v0-20260531-231355/checkpoint-165}"
M2_ROOT="${M2_ROOT:-outputs/runs/vlm/vlm-sft-qwen3vl4b-hermes-long-direct-blind-rag-1to1-e5-v2/formal618-native-async-20260819-130300/artifacts}"
RUN_ID="${RUN_ID:-m1-current-direct-bridge-$(date +%Y%m%d-%H%M%S)}"
ROOT="outputs/runs/vlm/vlm-direct-m1-current-bridge/$RUN_ID"
OUT="$ROOT/artifacts/m1_direct"
SERVICE="$ROOT/services/m1_direct"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"; export CUDA_VISIBLE_DEVICES
TP_SIZE="${SGLANG_TP_SIZE:-1}"; DP_SIZE="${SGLANG_DP_SIZE:-8}"
[[ -f "$MANIFEST" && -d "$M1_CHECKPOINT" ]] || { echo "missing M1 checkpoint or manifest" >&2; exit 2; }
for reference in "$M2_ROOT/raw_base_direct/scored.jsonl" "$M2_ROOT/candidate_direct/scored.jsonl"; do
  [[ -f "$reference" ]] || { echo "missing M2 Direct reference: $reference" >&2; exit 2; }
done
mkdir -p "$ROOT/logs" "$OUT" "$SERVICE"

validate() {
  [[ -f "$OUT/predictions.jsonl" && -f "$OUT/metrics.json" ]] || return 1
  "$PYTHON_BIN" vlm/eval/tools/normalize_answers.py --manifest "$MANIFEST" --predictions "$OUT/predictions.jsonl" --output-jsonl "$OUT/scored.jsonl" --output-metrics "$OUT/metrics.json" --output-csv "$OUT/scored.csv" >/dev/null
}

if ! validate; then
  rm -f "$SERVICE/service_status.json" "$SERVICE/service_exit.json" "$SERVICE/service.pid"
  setsid "$PYTHON_BIN" vlm/eval/tools/sglang_service.py --model-path "$M1_CHECKPOINT" --served-model-name agrinet-m1-current-direct --run-dir "$SERVICE" --tp-size "$TP_SIZE" --dp-size "$DP_SIZE" --max-running-requests 64 >"$SERVICE/launcher.log" 2>&1 &
  manager=$!
  api_base=""
  for _ in {1..420}; do
    [[ -f "$SERVICE/service_status.json" ]] && { api_base=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["api_base"])' "$SERVICE/service_status.json"); break; }
    kill -0 "$manager" 2>/dev/null || { tail -n 80 "$SERVICE/launcher.log" >&2; exit 1; }
    sleep 2
  done
  [[ -n "$api_base" ]] || { kill "$manager" 2>/dev/null || true; echo "service health timeout" >&2; exit 1; }
  args=(--manifest "$MANIFEST" --output "$OUT/predictions.jsonl" --repo-root "$REPO_ROOT" --model agrinet-m1-current-direct --api-base "$api_base" --max-concurrent 64 --request-retries 2 --snapshot-every 1 --max-new-tokens 512)
  [[ -f "$OUT/predictions.jsonl.run.json" ]] && args+=(--resume)
  set +e; "$PYTHON_BIN" vlm/eval/tools/run_qwen3_vl_direct_sglang_eval.py "${args[@]}"; rc=$?; kill -TERM "$manager" 2>/dev/null || true; wait "$manager" 2>/dev/null; set -e
  (( rc == 0 )) || exit "$rc"
  validate
fi

"$PYTHON_BIN" src/agrinet/vlm/evaluation/paired_bootstrap.py --candidate "$OUT/scored.jsonl" --baseline "$M2_ROOT/raw_base_direct/scored.jsonl" --out "$ROOT/artifacts/m1_vs_raw_base_direct_paired_review.json" --bootstrap-samples 10000 --seed 20260819
"$PYTHON_BIN" src/agrinet/vlm/evaluation/paired_bootstrap.py --candidate "$OUT/scored.jsonl" --baseline "$M2_ROOT/candidate_direct/scored.jsonl" --out "$ROOT/artifacts/m1_vs_m2_direct_paired_review.json" --bootstrap-samples 10000 --seed 20260819
"$PYTHON_BIN" - "$ROOT" "$MANIFEST" "$M1_CHECKPOINT" "$M2_ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root, manifest, checkpoint, m2_root = map(Path, sys.argv[1:])
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
summary = {
  'schema_version': 'agrinet.m1-current-direct-bridge/v1',
  'manifest': str(manifest), 'manifest_sha256': digest(manifest), 'rows': 618,
  'm1_checkpoint': str(checkpoint), 'm2_artifact_root': str(m2_root),
  'parallelism': {'backend': 'sglang.launch_server', 'tensor_parallel_size': 1, 'data_parallel_size': 8},
  'm1_metrics': json.loads((root/'artifacts/m1_direct/metrics.json').read_text()),
  'm1_predictions_sha256': digest(root/'artifacts/m1_direct/predictions.jsonl'),
  'm1_vs_raw_base': json.loads((root/'artifacts/m1_vs_raw_base_direct_paired_review.json').read_text()),
  'm1_vs_m2_candidate': json.loads((root/'artifacts/m1_vs_m2_direct_paired_review.json').read_text()),
}
(root/'artifacts/summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'rows': 618, 'm1_vs_raw_base_pp': summary['m1_vs_raw_base']['paired_delta_pp'], 'm1_vs_m2_pp': summary['m1_vs_m2_candidate']['paired_delta_pp']}))
PY
