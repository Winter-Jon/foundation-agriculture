#!/usr/bin/env bash
set -euo pipefail
repo_root=/data/home/jiangwentao/Repos/foundation-agriculture
cd "$repo_root"
run_dir=outputs/runs/rag/rag-sft-round115-stop-correction-pilot/logs
mkdir -p "$run_dir"
source <("$HOME/.apikeys/bin/apikey" env yunwu)
export YUNWU_API_BASE_URL=https://api3.wlai.vip/v1
.venv/bin/python -m agrinet.rag.distill.run_pilot \
  --plan-file outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round115_stop_correction_pilot/plan.jsonl \
  --candidate-source outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round115_stop_correction_pilot/source.jsonl \
  --limit 4 --rag-api http://127.0.0.1:8077 \
  --output-dir outputs/experiments/rag_sft_iteration/candidates/round115-stop-correction-pilot/pilot \
  --model gpt-5.6-luna --top-k 5 --max-tool-turns 3 --max-concurrent 1 \
  --teacher-timeout 180 --teacher-retries 2 --image-max-side 512 \
  >"$run_dir/pilot.log" 2>&1
printf '%s\n' "$?" >"$run_dir/exit_code"
