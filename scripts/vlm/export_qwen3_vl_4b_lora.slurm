#!/bin/bash
#SBATCH --job-name=qwen3vl-export-lora
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="/home/jiangwentao/Repos/foundation-agriculture"
cd "$REPO_ROOT"

source /etc/profile.d/modules.sh
module purge
module load cuda/12.8

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${SLURM_JOB_GPUS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
fi

export PYTHONPATH="$REPO_ROOT/vlm/sft/ms-swift:$REPO_ROOT:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
SWIFT_CMD=("$PYTHON_BIN" -m swift.cli.main)
ADAPTERS="${ADAPTERS:?Set ADAPTERS to an ms-swift LoRA checkpoint directory, e.g. outputs/vlm_sft/.../vx-*/checkpoint-*}"
OUTPUT_DIR="${OUTPUT_DIR:-${ADAPTERS%/}-merged}"

"${SWIFT_CMD[@]}" export \
    --adapters "$ADAPTERS" \
    --output_dir "$OUTPUT_DIR" \
    --merge_lora true

echo "Merged checkpoint: $OUTPUT_DIR"
