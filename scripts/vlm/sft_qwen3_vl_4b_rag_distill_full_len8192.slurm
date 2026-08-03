#!/bin/bash
#SBATCH --job-name=qwen3vl-sft-rag8192
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=192G
#SBATCH --time=18:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -r /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
    module purge
    module load cuda/12.8
fi

export WANDB_MODE=offline
export PYTHONPATH="$REPO_ROOT/vlm/sft/ms-swift:$REPO_ROOT:${PYTHONPATH:-}"
export QWENVL_BBOX_FORMAT=new
export MAX_PIXELS="${MAX_PIXELS:-1003520}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_ROOT/tmp/triton-cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${SLURM_JOB_GPUS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
fi

echo "HOSTNAME=$HOSTNAME"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MAX_PIXELS=$MAX_PIXELS"
echo "CONFIG=${CONFIG:-}"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
nvidia-smi -L || true

DATASET="outputs/vlm_sft/rag_distill/agent_sft.student.dedup.jsonl"
CONFIG="${CONFIG:-vlm/sft/configs/qwen3_vl_4b_rag_distill_student_e5_len8192_liger_lr1e5.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
SWIFT_CMD=("$PYTHON_BIN" -m swift.cli.main)

if [[ ! -f "$DATASET" ]]; then
    echo "RAG distill student dataset is missing: $DATASET" >&2
    exit 2
fi

mkdir -p "$TRITON_CACHE_DIR"

NPROC_PER_NODE=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | wc -l)
export NPROC_PER_NODE

"${SWIFT_CMD[@]}" sft "$CONFIG"
