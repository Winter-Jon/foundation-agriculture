#!/usr/bin/env bash
set -euo pipefail
ROOT=outputs/runs/vlm/m1-latest-ms-swift-queue-v1
PYTHON_BIN=.venv_test/bin/python
CANONICAL_PYTHON=.venv/bin/python
MANIFEST=outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl
M1=outputs/vlm_sft/qwen3_vl_4b_disease_pest_full_all_e5_len2048_liger_lr1e5_final/v0-20260531-231355/checkpoint-165
wait_for() {
  local name=$1 state
  while true; do
    state=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$ROOT/status/$name.json" 2>/dev/null || true)
    [[ $state == completed ]] && return
    [[ $state == failed ]] && { echo "$name failed; evaluation blocked" >&2; exit 20; }
    sleep 20
  done
}
wait_for historical
wait_for v3
for spec in historical:33:1 v3:64:1 historical:99:3 v3:192:3 historical:165:5 v3:320:5; do
  IFS=: read -r name step epoch <<<"$spec"
  d=$(cat "$ROOT/status/$name.train_dir")
  c="$d/checkpoint-$step"
  test -d "$c" || { echo "missing candidate $c" >&2; exit 21; }
  out="$ROOT/evaluations/$name/epoch-$epoch-checkpoint-$step"
  mkdir -p "$out"
  # A completed formal evaluation is immutable evidence.  Validate its summary
  # and retain it when this fail-closed queue is resumed after an interruption.
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
    echo "resume: validated completed evaluation $name epoch $epoch"
    continue
  fi
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CANDIDATE="$c" M1="$M1" MANIFEST="$MANIFEST" FORMAL_ROOT="$out" EXPERIMENT_ID="m1-latest-ms-swift-$name-e$epoch" SGLANG_TP_SIZE=1 SGLANG_DP_SIZE=8 MAX_NEW_TOKENS=2048 bash scripts/vlm/run_direct_formal618_native_dp8.sh >"$ROOT/logs/eval-$name-e$epoch.log" 2>&1
  "$CANONICAL_PYTHON" - "$out/artifacts/summary.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['rows']==618
for route in x['routes'].values():
    metrics = route['metrics']
    assert metrics['overall']['count'] == 618
    assert metrics['overall']['unparseable_rate'] == 0
    assert metrics['protocol_errors']['count'] == 0
PY
done
