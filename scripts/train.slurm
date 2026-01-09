#!/bin/bash
#SBATCH --job-name=agrinet-resnet50      # 作业名称
#SBATCH --partition=batch           # 指定GPU分区（根据集群调整）
#SBATCH --nodes=1                   # 节点数
#SBATCH --ntasks-per-node=1         # 每节点核心数
#SBATCH --cpus-per-task=32           # 每个任务需要的CPU核心
#SBATCH --gres=gpu:4                # 申请的GPU数量（重要！）
#SBATCH --mem=64G                   # 内存请求
#SBATCH --time=48:00:00             # 运行时间 (HH:MM:SS)
#SBATCH --output=slurm/%x_%j.out          # 输出日志 (%x=作业名, %j=作业ID)
#SBATCH --error=slurm/%x_%j.err           # 错误日志

source /etc/profile.d/modules.sh
module purge                        # 清理旧模块
module load cuda/12.8

export WANDB_API_KEY=c1d9f68903d5e8524b9a49600d08a71c26c8aa67
export WANDB_MODE=offline

export http_proxy="http://10.10.10.10:7899"
export https_proxy="http://10.10.10.10:7899"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

CONFIG=train/configs/base.yaml
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29502}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

OMP_NUM_THREADS=16

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    train/train.py \
    --config $CONFIG
