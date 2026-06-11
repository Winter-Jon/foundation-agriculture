#!/bin/bash
#SBATCH --job-name=qwen3vl-sft-full
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="/home/jiangwentao/Repos/foundation-agriculture"
cd "$REPO_ROOT"

source /etc/profile.d/modules.sh
module purge
module load cuda/12.8

export WANDB_MODE=offline
export PYTHONPATH="$REPO_ROOT/vlm/sft/ms-swift:$REPO_ROOT:${PYTHONPATH:-}"
export QWENVL_BBOX_FORMAT=new
export MAX_PIXELS="${MAX_PIXELS:-1003520}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${SLURM_JOB_GPUS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
fi

echo "HOSTNAME=$HOSTNAME"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L || true

if [[ ! -f outputs/vlm_data/disease_pest_large/sft_messages_en.jsonl || ! -f outputs/vlm_data/disease_pest_large/sft_messages_zh.jsonl ]]; then
    echo "Large SFT messages are missing." >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
SWIFT_CMD=("$PYTHON_BIN" -m swift.cli.main)

"$PYTHON_BIN" vlm/sft/tools/merge_sft_messages.py \
    --inputs \
        outputs/vlm_data/disease_pest_large/sft_messages_en.jsonl \
        outputs/vlm_data/disease_pest_large/sft_messages_zh.jsonl \
    --output outputs/vlm_sft/disease_pest_large/sft_messages_en_zh.jsonl \
    --repo-root "$REPO_ROOT" \
    --dedupe

NPROC_PER_NODE=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | wc -l)
export NPROC_PER_NODE

"${SWIFT_CMD[@]}" sft vlm/sft/configs/qwen3_vl_4b_disease_pest_large_full.yaml
