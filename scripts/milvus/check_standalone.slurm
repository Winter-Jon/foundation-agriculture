#!/bin/bash
#SBATCH --job-name=milvus-check-standalone
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="/home/jiangwentao/Repos/foundation-agriculture"
cd "$REPO_ROOT"

source /etc/profile.d/modules.sh
module purge
module load cuda/12.8

MILVUS_HOST="${MILVUS_HOST:-127.0.0.1}"
MILVUS_PORT="${MILVUS_PORT:-19530}"
MILVUS_URI="${MILVUS_URI:-http://${MILVUS_HOST}:${MILVUS_PORT}}"

echo "HOSTNAME=$HOSTNAME"
echo "MILVUS_URI=$MILVUS_URI"
"$REPO_ROOT/.venv/bin/python" tools/milvus/check_milvus_ready.py \
    --uri "$MILVUS_URI" \
    --timeout 20 \
    --interval 5
