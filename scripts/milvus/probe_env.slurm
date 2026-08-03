#!/bin/bash
#SBATCH --job-name=milvus-probe-env
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="/home/jiangwentao/Repos/foundation-agriculture"
cd "$REPO_ROOT"

source /etc/profile.d/modules.sh
module purge
module load cuda/12.8

echo "HOSTNAME=$HOSTNAME"
echo "PATH=$PATH"
echo "UV=$(command -v uv || true)"
echo "DOCKER=$(command -v docker || true)"
echo "PYTHON=$REPO_ROOT/.venv/bin/python"
nvidia-smi -L || true

if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    "$REPO_ROOT/.venv/bin/python" - <<'PY'
import importlib.util
import sys
mods = ["pymilvus", "milvus_lite", "torch", "transformers", "PIL", "sentencepiece"]
print("python", sys.version)
print("executable", sys.executable)
for mod in mods:
    print(f"module {mod}: {importlib.util.find_spec(mod)}")
PY
fi

echo "huggingface cache candidates"
find /home/jiangwentao/.cache/huggingface/hub -maxdepth 1 -type d -name 'models--*siglip*' -print || true

if command -v docker >/dev/null 2>&1; then
    docker ps --format '{{.Names}} {{.Image}} {{.Ports}}' || true
fi
