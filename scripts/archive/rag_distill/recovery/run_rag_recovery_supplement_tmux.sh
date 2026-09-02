#!/usr/bin/env bash
set -euo pipefail
cd /data/home/jiangwentao/Repos/foundation-agriculture
pilot=${PILOT_DIR:?PILOT_DIR required}
plan=${PLAN_FILE:?PLAN_FILE required}
name=${SUPPLEMENT_NAME:-supplement-1}
pilot=$(realpath "$pilot")
plan=$(realpath "$plan")
mkdir -p "$pilot/candidates/$name" "$pilot/review"
export PILOT_DIR="$pilot" PLAN_FILE="$plan" SUPPLEMENT_NAME="$name"
exec .venv/bin/python - <<'PY' >"$pilot/review/$name.log" 2>&1
import os,subprocess
from agrinet.common.credentials import yunwu_environment
from agrinet.common.network import local_proxy_environment
env=dict(os.environ)
if not env.get('YUNWU_API_KEY'): env.update(yunwu_environment())
if not any(env.get(k) for k in ('ALL_PROXY','HTTPS_PROXY','HTTP_PROXY')): env.update(local_proxy_environment())
p=os.environ['PILOT_DIR']; plan=os.environ['PLAN_FILE']; name=os.environ['SUPPLEMENT_NAME']
cmd=['.venv/bin/python','-m','agrinet.rag.distill.run_pilot','--plan-file',plan,'--candidate-source','outputs/vlm_data/disease_pest_large/contrast_samples_vit_base.jsonl','--limit','100000','--rag-api','http://127.0.0.1:8077','--output-dir',f'{p}/candidates/{name}','--model','gpt-5.6-luna','--max-tool-turns','3','--max-concurrent','4','--teacher-timeout','180','--teacher-retries','3']
subprocess.run(cmd,env=env,check=True)
selection=subprocess.run(['.venv/bin/python','-m','agrinet.rag.distill.select_recovery_pilot','--pilot-dir',p])
subprocess.run(['.venv/bin/python','-m','agrinet.rag.distill.validate_semantic_quality','--train-file',f'{p}/accepted/pilot_train_review.jsonl','--report',f'{p}/review/semantic_validation.json'])
raise SystemExit(selection.returncode)
PY
