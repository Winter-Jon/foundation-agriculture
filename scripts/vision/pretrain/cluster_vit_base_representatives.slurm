#!/bin/bash
#SBATCH --job-name=agrinet-vit-cluster
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

source /etc/profile.d/modules.sh
module purge
module load cuda/12.8

export WANDB_MODE=offline
export http_proxy="http://10.10.10.10:7899"
export https_proxy="http://10.10.10.10:7899"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
PORT=${PORT:-29512}

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    vision/pretrain/cluster_representatives.py \
    --data-dir datasets/AgriNet-1K/all \
    --checkpoint outputs/vit-base/vit-base-1/last.pth.tar \
    --model vit_base_patch16_224.orig_in21k_ft_in1k \
    --num-classes 1020 \
    --clusters-per-class 10 \
    --batch-size 512 \
    --workers 8 \
    --device cuda \
    --amp \
    --channels-last \
    --pin-mem \
    --features-dtype float16 \
    --out-dir outputs/vit-base-cluster/vit-base-1
