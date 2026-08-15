#!/usr/bin/env bash
set -euo pipefail
cd /data/home/jiangwentao/Repos/foundation-agriculture
set -o allexport
source <("$HOME/.apikeys/bin/apikey" env yunwu)
set +o allexport
sourcepool=outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round016_balanced_targets.jsonl
base=outputs/experiments/rag_sft_iteration/candidates
run_one() {
  local name="$1" plan="$2"
  mkdir -p "$base/$name"
  YUNWU_API_BASE_URL=https://api3.wlai.vip/v1 PYTHONPATH=. .venv/bin/python -m tools.rag_distill.run_pilot \
    --plan-file "outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/$plan" \
    --candidate-source "$sourcepool" --limit 1 --rag-api http://127.0.0.1:8077 \
    --output-dir "$base/$name" --model gpt-4o --max-tool-turns 3 --top-k 5 \
    --max-concurrent 1 --temperature 0 --max-tokens 512 --teacher-timeout 180 \
    --teacher-retries 2 --teacher-retry-sleep 2 --reasoning-effort none
}
run_one round016-api3-gpt4o-en-disease round016_en-disease_N04063_P00001_plan.jsonl
run_one round016-api3-gpt4o-zh-disease round016_zh-disease_N04029_P00001_plan.jsonl
run_one round016-api3-gpt4o-zh-pest round016_zh-pest_N05070_P00006_plan.jsonl
