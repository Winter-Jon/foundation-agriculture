#!/bin/bash
#SBATCH --job-name=pack-agrinet1k-open
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load cuda/12.8

cd /home/jiangwentao/Repos/foundation-agriculture

/home/jiangwentao/dev/miniconda/envs/torch/bin/python tools/create_agrinet1k_open_domain_split.py \
  --pack-only \
  --split-info datasets/AgriNet-1K/open_domain/manifests/split_info_20260527_145700_strict.json \
  --pack-workers "${SLURM_CPUS_PER_TASK:-32}" \
  --shard-size 2000 \
  --target-size 256
