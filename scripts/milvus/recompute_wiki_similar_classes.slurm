#!/bin/bash
#SBATCH --job-name=wiki-similar-siglip2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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
export http_proxy="${http_proxy:-http://10.10.10.10:7899}"
export https_proxy="${https_proxy:-http://10.10.10.10:7899}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"

MODEL_NAME="${MODEL_NAME:-models/siglip2-so400m-patch16-naflex}"
KNOWLEDGE_BASE="${KNOWLEDGE_BASE:-datasets/AgriNet-1K/wiki/base.json}"
IMAGE_DIR="${IMAGE_DIR:-datasets/AgriNet-1K/wiki/images}"
REPORT="${REPORT:-outputs/milvus/wiki_similar_classes_siglip2_report.json}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TOP_K="${TOP_K:-5}"

echo "HOSTNAME=$HOSTNAME"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MODEL_NAME=$MODEL_NAME"
echo "KNOWLEDGE_BASE=$KNOWLEDGE_BASE"
echo "IMAGE_DIR=$IMAGE_DIR"
echo "REPORT=$REPORT"
nvidia-smi -L || true

mkdir -p outputs/milvus slurm

"$REPO_ROOT/.venv/bin/python" src/agrinet/rag/milvus_tools/recompute_wiki_similar_classes.py \
    --knowledge-base "$KNOWLEDGE_BASE" \
    --image-dir "$IMAGE_DIR" \
    --model-name "$MODEL_NAME" \
    --report "$REPORT" \
    --batch-size "$BATCH_SIZE" \
    --top-k "$TOP_K" \
    --device cuda \
    --write
