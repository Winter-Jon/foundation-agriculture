#!/bin/bash
#SBATCH --job-name=prep-dp-train-visible-r5
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="/home/jiangwentao/Repos/foundation-agriculture"
cd "$REPO_ROOT"

source /etc/profile.d/modules.sh
module purge

PYTHON_BIN="${PYTHON_BIN:-/home/jiangwentao/dev/miniconda/envs/torch/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-outputs/vlm_data/disease_pest_train_visible_random5}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-5}"
NEGATIVE_COUNT="${NEGATIVE_COUNT:-5}"
RANDOM_SEED="${RANDOM_SEED:-42}"
SPLIT_INFO="${SPLIT_INFO:-datasets/AgriNet-1K/open_domain/manifests/split_info_20260527_145700_strict.json}"

echo "PYTHON_BIN=$PYTHON_BIN"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "SAMPLES_PER_CLASS=$SAMPLES_PER_CLASS"
echo "NEGATIVE_COUNT=$NEGATIVE_COUNT"
echo "RANDOM_SEED=$RANDOM_SEED"
echo "SPLIT_INFO=$SPLIT_INFO"

"$PYTHON_BIN" tools/vloom_agrinet/prepare_disease_pest_data.py \
  --data-root datasets/AgriNet-1K \
  --feature-root outputs/vit-base-cluster/vit-base-1 \
  --output-dir "$OUTPUT_DIR" \
  --samples-per-class "$SAMPLES_PER_CLASS" \
  --negative-count "$NEGATIVE_COUNT" \
  --reuse-pairs outputs/vlm_data/disease_pest/finegrained_pairs_vit_base.jsonl \
  --train-split-info "$SPLIT_INFO" \
  --random-seed "$RANDOM_SEED"

"$PYTHON_BIN" tools/vloom_agrinet/validate_disease_pest_data.py \
  --data-dir "$OUTPUT_DIR" \
  --expected-disease 79 \
  --expected-pest 61 \
  --top-k 5 \
  --expected-negatives "$NEGATIVE_COUNT" \
  --samples-per-class "$SAMPLES_PER_CLASS"

echo "Prepared train-visible random samples under $OUTPUT_DIR"
