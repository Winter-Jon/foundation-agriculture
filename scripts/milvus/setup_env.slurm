#!/bin/bash
#SBATCH --job-name=milvus-setup-env
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="/home/jiangwentao/Repos/foundation-agriculture"
cd "$REPO_ROOT"

source /etc/profile.d/modules.sh
module purge
module load cuda/12.8

export HF_HOME="${HF_HOME:-/home/jiangwentao/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export MODEL_NAME="${MODEL_NAME:-models/siglip2-so400m-patch16-naflex}"
export http_proxy="${http_proxy:-http://10.10.10.10:7899}"
export https_proxy="${https_proxy:-http://10.10.10.10:7899}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"

echo "HOSTNAME=$HOSTNAME"
echo "PYTHON=$REPO_ROOT/.venv/bin/python"
echo "MODEL_NAME=$MODEL_NAME"
echo "WHEEL_DIR=${WHEEL_DIR:-}"
nvidia-smi -L || true

if [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
    uv venv "$REPO_ROOT/.venv"
fi

if "$REPO_ROOT/.venv/bin/python" - <<'PY'
import importlib
for mod in ["pymilvus", "milvus_lite", "torch", "transformers", "PIL"]:
    importlib.import_module(mod)
print("required_imports_present")
PY
then
    echo "Python dependencies already present; skipping installation."
else
    if [ -n "${WHEEL_DIR:-}" ]; then
        uv pip install -p "$REPO_ROOT/.venv/bin/python" --no-index --find-links "$WHEEL_DIR" \
            'pymilvus[milvus_lite]==2.5.4' \
            'transformers>=4.53.0' \
            torch \
            pillow \
            protobuf \
            sentencepiece
    else
        uv pip install -p "$REPO_ROOT/.venv/bin/python" \
            'pymilvus[milvus_lite]==2.5.4' \
            'transformers>=4.53.0' \
            torch \
            pillow \
            protobuf \
            sentencepiece
    fi
fi

"$REPO_ROOT/.venv/bin/python" - <<'PY'
import importlib
mods = ["pymilvus", "torch", "transformers", "PIL"]
for mod in mods:
    importlib.import_module(mod)
print("imports_ok")
PY

"$REPO_ROOT/.venv/bin/python" - <<'PY'
import os
from transformers import AutoModel, AutoProcessor
model_name = os.environ["MODEL_NAME"]
print(f"preloading_model={model_name}")
AutoProcessor.from_pretrained(model_name)
AutoModel.from_pretrained(model_name)
print("model_preload_ok")
PY
