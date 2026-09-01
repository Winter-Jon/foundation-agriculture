#!/usr/bin/env bash
# Formal Hermes-v2 evaluation with one native SGLang TP=1, DP=8 service per route.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
MANIFEST="outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl"
CANDIDATE="outputs/vlm_sft/qwen3_vl_4b_hermes_long_direct_blind_rag_1to1_e5/v0-20260819-075722/checkpoint-175"
ROOT="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/evaluation/formal618-native-dp8"
LOG="outputs/runs/vlm/vlm-sft-qwen3vl4b-hermes-long-direct-blind-rag-1to1-e5-v2/formal618-native-dp8.log"
mkdir -p "$ROOT" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

[[ -f "$MANIFEST" && -d "$CANDIDATE" ]] || { echo "missing manifest or candidate" >&2; exit 2; }
for route in candidate_direct raw_base_direct candidate_rag raw_base_rag; do
  [[ ! -e "$ROOT/$route/metrics.json" && ! -e "$ROOT/$route/run_status.exit_code" && ! -e "$ROOT/$route/predictions.jsonl" ]] || { echo "refusing output reuse: $ROOT/$route" >&2; exit 2; }
done

run_direct() {
  local variant=$1 model=$2
  echo "[$(date --iso-8601=seconds)] native DP8 Direct $variant"
  MODEL_PATH="$model" MANIFEST="$MANIFEST" OUT_DIR="$ROOT/${variant}_direct" \
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" SGLANG_TP_SIZE=1 SGLANG_DP_SIZE=8 \
    SGLANG_MEM_FRACTION_STATIC=0.7 MAX_NEW_TOKENS=512 REQUEST_TIMEOUT=1200 \
    bash scripts/vlm/run_local_direct_retention_eval.sh
}
run_rag() {
  local variant=$1 model=$2
  echo "[$(date --iso-8601=seconds)] native DP8 RAG $variant"
  MODEL_PATH="$model" MANIFEST="$MANIFEST" OUT_DIR="$ROOT/${variant}_rag" \
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" SGLANG_TP_SIZE=1 SGLANG_DP_SIZE=8 \
    SGLANG_MEM_FRACTION_STATIC=0.7 MAX_NEW_TOKENS=512 MAX_TOOL_TURNS=3 TOP_K=3 REQUEST_TIMEOUT=1200 \
    DISABLE_FORCED_FIRST_CALL=1 bash scripts/vlm/run_local_rag_sft_eval.sh
}

run_direct candidate "$CANDIDATE"
run_direct raw_base models/Qwen3-VL-4B-Instruct
run_rag candidate "$CANDIDATE"
run_rag raw_base models/Qwen3-VL-4B-Instruct

for route in candidate_direct raw_base_direct candidate_rag raw_base_rag; do
  [[ "$(tr -d '[:space:]' <"$ROOT/$route/run_status.exit_code")" == 0 ]] || { echo "nonzero route: $route" >&2; exit 1; }
done
[[ ! -e "$ROOT/candidate_vs_base_direct_paired_review.json" && ! -e "$ROOT/candidate_vs_base_rag_paired_review.json" && ! -e "$ROOT/summary.json" ]] || { echo "refusing review output reuse" >&2; exit 2; }
.venv/bin/python src/agrinet/rag/distill/review_matched_diagnostic.py --candidate "$ROOT/candidate_direct/scored.jsonl" --baseline "$ROOT/raw_base_direct/scored.jsonl" --out "$ROOT/candidate_vs_base_direct_paired_review.json" --bootstrap-samples 10000 --seed 20260819
.venv/bin/python src/agrinet/rag/distill/review_matched_diagnostic.py --candidate "$ROOT/candidate_rag/scored.jsonl" --baseline "$ROOT/raw_base_rag/scored.jsonl" --out "$ROOT/candidate_vs_base_rag_paired_review.json" --bootstrap-samples 10000 --seed 20260819
.venv/bin/python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
summary={
  'schema_version':'agrinet.hermes-v2-formal618-native-dp8-summary/v1',
  'manifest':'outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl',
  'manifest_sha256':'146cc671daa498d4fa91001b6fd7e0c431c30ede0526940e92d82d3cba40e82f',
  'rows':618, 'parallelism':{'backend':'sglang','tensor_parallel_size':1,'data_parallel_size':8},
  'candidate_checkpoint':'outputs/vlm_sft/qwen3_vl_4b_hermes_long_direct_blind_rag_1to1_e5/v0-20260819-075722/checkpoint-175',
  'direct':json.loads((root/'candidate_vs_base_direct_paired_review.json').read_text()),
  'rag':json.loads((root/'candidate_vs_base_rag_paired_review.json').read_text()),
}
(root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'rows':618,'direct_delta_pp':summary['direct']['paired_delta_pp'],'rag_delta_pp':summary['rag']['paired_delta_pp']}))
PY
