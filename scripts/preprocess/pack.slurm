#!/bin/bash
#SBATCH --job-name=pack      # 作业名称
#SBATCH --partition=batch           # 指定GPU分区（根据集群调整）
#SBATCH --nodes=1                   # 节点数
#SBATCH --ntasks-per-node=1         # 每节点核心数
#SBATCH --cpus-per-task=32           # 每个任务需要的CPU核心
#SBATCH --gres=gpu:0                # 申请的GPU数量（重要！）
#SBATCH --mem=64G                   # 内存请求
#SBATCH --time=8:00:00             # 运行时间 (HH:MM:SS)
#SBATCH --output=slurm/%x_%j.out          # 输出日志 (%x=作业名, %j=作业ID)
#SBATCH --error=slurm/%x_%j.err           # 错误日志

source /etc/profile.d/modules.sh
module purge                        # 清理旧模块
module load cuda/12.8

export http_proxy="http://10.10.10.10:7899"
export https_proxy="http://10.10.10.10:7899"

# python preprocess/prepare_wds_folds.py
python preprocess/split_wds.py
