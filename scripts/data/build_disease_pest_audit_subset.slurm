#!/bin/bash
#SBATCH --job-name=build-dp-audit-v1
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge

REPO_ROOT=/data/home/jiangwentao/Repos/foundation-agriculture
DATASET_ROOT=/data/share/datasets/agriculture/AgriNet-1K
OUTPUT_ROOT=${OUTPUT_ROOT:-$DATASET_ROOT/open_agri_v1}

cd "$REPO_ROOT"
python3 scripts/data/build_disease_pest_audit_subset.py \
  --dataset-root "$DATASET_ROOT" \
  --output-root "$OUTPUT_ROOT"
