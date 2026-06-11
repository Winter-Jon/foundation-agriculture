#!/bin/bash
#SBATCH --job-name=qwen3vl-sft-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
SWIFT_CMD=("$PYTHON_BIN" -m swift.cli.main)

"$PYTHON_BIN" vlm/sft/tools/merge_sft_messages.py \
    --inputs \
        outputs/vlm_data/disease_pest_test/sft_messages_en.jsonl \
        outputs/vlm_data/disease_pest_test/sft_messages_zh.jsonl \
    --output outputs/vlm_sft/disease_pest_test/sft_messages_en_zh.jsonl \
    --repo-root "$REPO_ROOT" \
    --dedupe

"${SWIFT_CMD[@]}" sft vlm/sft/configs/qwen3_vl_4b_disease_pest_smoke.yaml
