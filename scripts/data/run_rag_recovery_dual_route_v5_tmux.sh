#!/usr/bin/env bash
set -euo pipefail
cd /data/home/jiangwentao/Repos/foundation-agriculture
pilot=${PILOT_DIR:?PILOT_DIR required}
plan=${PLAN_FILE:?PLAN_FILE required}
name=${RUN_NAME:-dual-route-12x2}
pilot=$(realpath "$pilot")
plan=$(realpath "$plan")
output="$pilot/candidates/$name"
limit=${LIMIT:-$(wc -l < "$plan")}
mkdir -p "$output" "$pilot/review"
export PILOT_DIR="$pilot" PLAN_FILE="$plan" OUTPUT_DIR="$output" LIMIT="$limit"
exec .venv/bin/python - <<'PY'
import os, subprocess
from agrinet.common.credentials import yunwu_environment
from agrinet.common.network import local_proxy_environment
env=dict(os.environ)
if not env.get('YUNWU_API_KEY'): env.update(yunwu_environment())
if not any(env.get(k) for k in ('ALL_PROXY','HTTPS_PROXY','HTTP_PROXY')): env.update(local_proxy_environment())
p=os.environ['PILOT_DIR']; plan=os.environ['PLAN_FILE']; output=os.environ['OUTPUT_DIR']
limit=os.environ['LIMIT']
cmd=['.venv/bin/python','-m','tools.rag_distill.run_pilot','--plan-file',plan,
     '--candidate-source','outputs/vlm_data/disease_pest_large/contrast_samples_vit_base.jsonl',
     '--limit',limit,'--rag-api','http://127.0.0.1:8077','--output-dir',output,
     '--model','gpt-5.6-luna','--max-tool-turns','3','--max-concurrent','4',
     '--teacher-timeout','180','--teacher-retries','3']
subprocess.run(cmd,env=env,check=True)
subprocess.run(['.venv/bin/python','-m','tools.rag_distill.validate_artifact','--artifact-dir',output,
                '--min-retrieval-success-rate','1.0'],check=True)
subprocess.run(['.venv/bin/python','-m','tools.rag_distill.validate_semantic_quality',
                '--train-file',f'{output}/train/agent_sft.accepted.jsonl',
                '--report',f'{output}/reports/semantic_validation.json'],check=True)
PY
