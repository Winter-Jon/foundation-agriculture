#!/bin/bash
#SBATCH --job-name=milvus-ingest-wiki
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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

MILVUS_MODE="${MILVUS_MODE:-lite}"
MILVUS_HOST="${MILVUS_HOST:-127.0.0.1}"
MILVUS_PORT="${MILVUS_PORT:-19530}"
MILVUS_URI="${MILVUS_URI:-http://${MILVUS_HOST}:${MILVUS_PORT}}"
MILVUS_LITE_DB="${MILVUS_LITE_DB:-outputs/milvus/agrinet_wiki_lite.db}"
COLLECTION="${COLLECTION:-agrinet_wiki_siglip2}"
CLASS_COLLECTION="${CLASS_COLLECTION:-}"
IMAGE_COLLECTION="${IMAGE_COLLECTION:-}"
MODEL_NAME="${MODEL_NAME:-models/siglip2-so400m-patch16-naflex}"
KNOWLEDGE_BASE="${KNOWLEDGE_BASE:-datasets/AgriNet-1K/wiki/base.json}"
IMAGE_DIR="${IMAGE_DIR:-datasets/AgriNet-1K/wiki/images}"
BATCH_SIZE="${BATCH_SIZE:-8}"

echo "HOSTNAME=$HOSTNAME"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MILVUS_MODE=$MILVUS_MODE"
echo "MILVUS_URI=$MILVUS_URI"
echo "MILVUS_LITE_DB=$MILVUS_LITE_DB"
echo "COLLECTION=$COLLECTION"
echo "CLASS_COLLECTION=${CLASS_COLLECTION:-${COLLECTION}_classes}"
echo "IMAGE_COLLECTION=${IMAGE_COLLECTION:-${COLLECTION}_images}"
echo "MODEL_NAME=$MODEL_NAME"
echo "KNOWLEDGE_BASE=$KNOWLEDGE_BASE"
echo "IMAGE_DIR=$IMAGE_DIR"
nvidia-smi -L || true

mkdir -p outputs/milvus slurm

if [ "$MILVUS_MODE" = "standalone" ]; then
    "$REPO_ROOT/.venv/bin/python" tools/milvus/check_milvus_ready.py \
        --uri "$MILVUS_URI" \
        --timeout 300 \
        --interval 10
fi

ARGS=(
    --mode "$MILVUS_MODE"
    --milvus-uri "$MILVUS_URI"
    --lite-db "$MILVUS_LITE_DB"
    --collection "$COLLECTION"
    --model-name "$MODEL_NAME"
    --knowledge-base "$KNOWLEDGE_BASE"
    --image-dir "$IMAGE_DIR"
    --batch-size "$BATCH_SIZE"
    --drop-existing
)
if [ -n "$CLASS_COLLECTION" ]; then
    ARGS+=(--class-collection "$CLASS_COLLECTION")
fi
if [ -n "$IMAGE_COLLECTION" ]; then
    ARGS+=(--image-collection "$IMAGE_COLLECTION")
fi

"$REPO_ROOT/.venv/bin/python" tools/milvus/ingest_agrinet_wiki.py "${ARGS[@]}"
