#!/usr/bin/env bash
set -euo pipefail
cd /data/home/jiangwentao/Repos/foundation-agriculture
set -o allexport
source <("$HOME/.apikeys/bin/apikey" env yunwu)
set +o allexport
sourcepool=outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round017_english_open_targets.jsonl
for spec in "round018-api3-gpt4o-en-disease round018_en-disease_N04130_P00001_plan.jsonl" "round018-api3-gpt4o-en-pest round018_en-pest_N05042_P00001_plan.jsonl"; do
  set -- $spec
  mkdir -p outputs/experiments/rag_sft_iteration/candidates/$1
  YUNWU_API_BASE_URL=https://api3.wlai.vip/v1 PYTHONPATH=. .venv/bin/python -m tools.rag_distill.run_pilot \
    --plan-file outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/$2 --candidate-source $sourcepool \
    --limit 1 --rag-api http://127.0.0.1:8077 --output-dir outputs/experiments/rag_sft_iteration/candidates/$1 \
    --model gpt-4o --max-tool-turns 3 --top-k 5 --max-concurrent 1 --temperature 0 --max-tokens 512 \
    --teacher-timeout 180 --teacher-retries 2 --teacher-retry-sleep 2 --reasoning-effort none
done
