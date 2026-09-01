#!/usr/bin/env bash
set -euo pipefail
ROOT=outputs/runs/vlm/m1-latest-ms-swift-queue-v1
PYTHON_BIN=.venv_test/bin/python
status() { "$PYTHON_BIN" - "$ROOT/status/v3.json" "$1" "$2" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
p,state,code=sys.argv[1:]; Path(p).write_text(json.dumps({'state':state,'exit_code':None if code=='null' else int(code),'updated_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
PY
}
status running null
set +e
CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC_PER_NODE=4 MASTER_PORT=29922 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON_BIN" -m swift.cli.main sft configs/vlm/qwen3_vl_4b_m1_latest_ms_swift_v3_4gpu_sft.yaml >"$ROOT/logs/v3.log" 2>&1
rc=$?
set -e
if (( rc )); then status failed "$rc"; exit "$rc"; fi
d=$(find outputs/vlm_sft/qwen3_vl_4b_m1_latest_ms_swift_v3_4gpu -mindepth 1 -maxdepth 1 -type d -name 'v0-*' -printf '%p\n' | sort | tail -1)
test -f "$d/checkpoint-320/trainer_state.json" || { status failed 99; exit 99; }
"$PYTHON_BIN" - "$d/checkpoint-320/trainer_state.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))['global_step']==320
PY
printf '%s\n' "$d" >"$ROOT/status/v3.train_dir"
status completed 0
