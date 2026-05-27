#!/bin/bash
#SBATCH --job-name=agrinet-mae-vit-base-1gpu
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu03
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

source /etc/profile.d/modules.sh
module purge
module load cuda/12.8

export WANDB_MODE=offline

echo "HOSTNAME=$HOSTNAME"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES(before)=${CUDA_VISIBLE_DEVICES:-}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${SLURM_JOB_GPUS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
fi

echo "CUDA_VISIBLE_DEVICES(after)=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L || true
python3 - <<'PY'
import os
import torch
print("torch_cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch_cuda_available", torch.cuda.is_available())
print("torch_cuda_device_count", torch.cuda.device_count())
PY

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

CONFIG=vision/pretrain/configs/mae_vit_base.yaml
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29504}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export OMP_NUM_THREADS=8

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    vision/pretrain/train_mae.py \
    --config $CONFIG \
    --batch-size 64 \
    --workers 8 \
    --train-num-samples 6400 \
    --max-train-steps 20 \
    --experiment mae-vit-base-smoke-v2-1gpu
