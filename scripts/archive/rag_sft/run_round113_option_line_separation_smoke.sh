#!/usr/bin/env bash
set -uo pipefail
cd /data/home/jiangwentao/Repos/foundation-agriculture
run_dir=outputs/runs/rag/rag-sft-round113-option-line-separation-smoke/logs
mkdir -p "$run_dir"
set -a
eval "$(/data/home/jiangwentao/.apikeys/bin/apikey env yunwu)"
set +a
export YUNWU_API_BASE_URL=https://api3.wlai.vip/v1
.venv/bin/python -m tools.rag_distill.run_pilot \
  --plan-file outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round113_option_line_separation_smoke/plan.jsonl \
  --candidate-source outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round113_option_line_separation_smoke/source.jsonl \
  --limit 1 --rag-api http://127.0.0.1:8077 \
  --output-dir outputs/experiments/rag_sft_iteration/candidates/round113-option-line-separation-smoke/pilot \
  --model gpt-5.6-luna --top-k 5 --max-tool-turns 3 --max-concurrent 1 \
  --teacher-timeout 120 --image-max-side 512 > "$run_dir/pilot.log" 2>&1
run_code=$?
printf '%s\n' "$run_code" > "$run_dir/pilot.exit_code"
exit "$run_code"
