#!/usr/bin/env bash
set -euo pipefail
ROOT=outputs/runs/vlm/vlm-sft-qwen3vl4b-m1-latest-ms-swift-v4-m1-system-format-8gpu-v1
CANONICAL_PYTHON=.venv/bin/python
MANIFEST=outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl
M1=outputs/vlm_sft/qwen3_vl_4b_disease_pest_full_all_e5_len2048_liger_lr1e5_final/v0-20260531-231355/checkpoint-165
TRAIN_DIR=outputs/vlm_sft/qwen3_vl_4b_m1_latest_ms_swift_v4_m1_system_format_8gpu/v0-20260901-163603
for spec in 64:1 192:3 320:5; do
  IFS=: read -r step epoch <<<"$spec"
  candidate="$TRAIN_DIR/checkpoint-$step"
  out="$ROOT/evaluations/epoch-$epoch-checkpoint-$step"
  test -d "$candidate" || { echo "missing candidate: $candidate" >&2; exit 21; }
  mkdir -p "$out" "$ROOT/logs"
  if [[ -f "$out/artifacts/summary.json" ]]; then
    "$CANONICAL_PYTHON" - "$out/artifacts/summary.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
assert x['rows'] == 618
assert x['manifest_sha256'] == '80d79f3ac4860d17ac3612265254240b79d29babf310c94d1c0b3b86d5fc0508'
for route in x['routes'].values():
    metrics = route['metrics']
    assert metrics['overall']['count'] == 618
    assert metrics['overall']['unparseable_rate'] == 0
    assert metrics['protocol_errors']['count'] == 0
PY
    echo "resume: validated epoch $epoch"
    continue
  fi
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CANDIDATE="$candidate" M1="$M1" MANIFEST="$MANIFEST" FORMAL_ROOT="$out" EXPERIMENT_ID="m1-latest-ms-swift-v4-m1-system-format-e$epoch" SGLANG_TP_SIZE=1 SGLANG_DP_SIZE=8 MAX_NEW_TOKENS=2048 bash scripts/vlm/run_direct_formal618_native_dp8.sh >"$ROOT/logs/eval-e$epoch.log" 2>&1
  "$CANONICAL_PYTHON" - "$out/artifacts/summary.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
assert x['rows'] == 618
assert x['manifest_sha256'] == '80d79f3ac4860d17ac3612265254240b79d29babf310c94d1c0b3b86d5fc0508'
for route in x['routes'].values():
    metrics = route['metrics']
    assert metrics['overall']['count'] == 618
    assert metrics['overall']['unparseable_rate'] == 0
    assert metrics['protocol_errors']['count'] == 0
PY
done
