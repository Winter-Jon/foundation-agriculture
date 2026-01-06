#!/bin/bash
#SBATCH --job-name=validate      # 作业名称
#SBATCH --partition=batch           # 指定GPU分区（根据集群调整）
#SBATCH --nodes=1                   # 节点数
#SBATCH --ntasks-per-node=1         # 每节点核心数
#SBATCH --cpus-per-task=4           # 每个任务需要的CPU核心
#SBATCH --gres=gpu:1                # 申请的GPU数量（重要！）
#SBATCH --mem=8G                   # 内存请求
#SBATCH --time=00:10:00             # 运行时间 (HH:MM:SS)
#SBATCH --output=slurm/%x_%j.out          # 输出日志 (%x=作业名, %j=作业ID)
#SBATCH --error=slurm/%x_%j.err           # 错误日志

source /etc/profile.d/modules.sh
module purge                        # 清理旧模块
module load cuda/12.8

export WANDB_API_KEY=c1d9f68903d5e8524b9a49600d08a71c26c8aa67
export WANDB_MODE=offline

export http_proxy="http://10.10.10.10:7899"
export https_proxy="http://10.10.10.10:7899"


python train/validate.py \
  -c train/configs/validate.yaml \
  --data-dir datasets/AgriNet-1K/wds_views/fold_3 \
  --checkpoint outputs/resnet50-hard-mining-1/resnet50_fold_3/model_best.pth.tar \
  --results-file outputs/resnet50-hard-mining-1/resnet50_fold_3/metrics.csv