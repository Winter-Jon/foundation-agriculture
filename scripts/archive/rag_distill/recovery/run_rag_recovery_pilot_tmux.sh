#!/usr/bin/env bash
set -euo pipefail
cd /data/home/jiangwentao/Repos/foundation-agriculture
pilot_dir=${PILOT_DIR:-outputs/artifacts/datasets/agrinet-rag-recovery-pilot-v2}
plan_file=${PLAN_FILE:-$pilot_dir/plan/rag_candidate_attempts.preflight.jsonl}
mkdir -p "$pilot_dir/candidates/batch" "$pilot_dir/review"
export PILOT_DIR="$pilot_dir"
export PLAN_FILE="$plan_file"
exec .venv/bin/python - <<'PY' >"$pilot_dir/review/tmux-pilot.log" 2>&1
import os
import subprocess
from agrinet.common.credentials import yunwu_environment
from agrinet.common.network import local_proxy_environment
env = dict(os.environ)
if not env.get('YUNWU_API_KEY'):
    env.update(yunwu_environment())
if not any(env.get(key) for key in ('ALL_PROXY', 'HTTPS_PROXY', 'HTTP_PROXY')):
    env.update(local_proxy_environment())
pilot = os.environ['PILOT_DIR']
plan = os.environ['PLAN_FILE']
command = ['.venv/bin/python','-m','agrinet.rag.distill.run_pilot',
    '--plan-file',plan,
    '--candidate-source','outputs/vlm_data/disease_pest_large/contrast_samples_vit_base.jsonl',
    '--limit','100000','--rag-api','http://127.0.0.1:8077','--output-dir',f'{pilot}/candidates/batch',
    '--model','gpt-5.6-luna','--max-tool-turns','3','--max-concurrent','4',
    '--teacher-timeout','180','--teacher-retries','3']
subprocess.run(command, env=env, check=True)
selection = subprocess.run(['.venv/bin/python','-m','agrinet.rag.distill.select_recovery_pilot','--pilot-dir',pilot])
final_file = f'{pilot}/accepted/pilot_train_review.jsonl'
if os.path.isfile(final_file):
    subprocess.run(['.venv/bin/python','-m','agrinet.rag.distill.validate_semantic_quality','--train-file',final_file,'--report',f'{pilot}/review/semantic_validation.json'])
raise SystemExit(selection.returncode)
PY
