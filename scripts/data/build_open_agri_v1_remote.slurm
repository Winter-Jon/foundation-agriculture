#!/bin/bash
#SBATCH --job-name=build-open-agri-v1
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

REMOTE_HOST=node39
REMOTE_REPO=/data/home/jiangwentao/Repos/foundation-agriculture

ssh -o BatchMode=yes "$REMOTE_HOST" \
  "cd '$REMOTE_REPO' && python3 scripts/data/build_disease_pest_audit_subset.py --dataset-root /data/share/datasets/agriculture/AgriNet-1K --output-root /data/share/datasets/agriculture/AgriNet-1K/open_agri_v1"
