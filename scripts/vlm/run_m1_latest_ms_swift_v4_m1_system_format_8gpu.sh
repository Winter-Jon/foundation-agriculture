#!/usr/bin/env bash
set -euo pipefail
ROOT=outputs/runs/vlm/vlm-sft-qwen3vl4b-m1-latest-ms-swift-v4-m1-system-format-8gpu-v1
PYTHON_BIN=.venv_test/bin/python
CONFIG=configs/vlm/qwen3_vl_4b_m1_latest_ms_swift_v4_m1_system_format_8gpu_sft.yaml
MODEL_ROOT=outputs/vlm_sft/qwen3_vl_4b_m1_latest_ms_swift_v4_m1_system_format_8gpu
mkdir -p "$ROOT/logs" "$ROOT/status"
write_status() {
  "$PYTHON_BIN" -c "import json,sys; from pathlib import Path; from datetime import datetime,timezone; Path(sys.argv[1]).write_text(json.dumps({'status':sys.argv[2],'exit_code':None if sys.argv[3]=='null' else int(sys.argv[3]),'updated_at':datetime.now(timezone.utc).isoformat()},indent=2)+chr(10))" "$ROOT/status/status.json" "$1" "$2"
}
test -x "$PYTHON_BIN" || { echo "missing test environment" >&2; exit 2; }
test -f "$CONFIG" || { echo "missing SFT config" >&2; exit 2; }
test ! -e "$MODEL_ROOT" || { echo "refusing to overwrite model output: $MODEL_ROOT" >&2; exit 3; }
write_status running null
set +e
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 MASTER_PORT=29941 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON_BIN" -m swift.cli.main sft "$CONFIG" >"$ROOT/logs/train.log" 2>&1
rc=$?
set -e
if (( rc )); then write_status failed "$rc"; exit "$rc"; fi
d=$(find "$MODEL_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'v0-*' -printf '%p\n' | sort | tail -1)
test -f "$d/checkpoint-320/trainer_state.json" || { write_status failed 99; exit 99; }
"$PYTHON_BIN" -c "import json,sys; assert json.load(open(sys.argv[1]))['global_step']==320" "$d/checkpoint-320/trainer_state.json"
printf '%s\n' "$d" >"$ROOT/status/train_dir"
write_status completed 0
