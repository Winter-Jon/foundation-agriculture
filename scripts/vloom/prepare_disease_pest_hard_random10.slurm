#!/bin/bash
#SBATCH --job-name=prep-dp-hard-r10
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

OUTPUT_DIR="${OUTPUT_DIR:-outputs/vlm_data/disease_pest_hard_random10_no_rep}"
CLASS_LIST="${CLASS_LIST:-outputs/vlm_data/disease_pest_hard_random10/hard_classes.txt}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-10}"
NEGATIVE_COUNT="${NEGATIVE_COUNT:-5}"
RANDOM_SEED="${RANDOM_SEED:-43}"
SPLIT_INFO="${SPLIT_INFO:-datasets/AgriNet-1K/open_domain/manifests/split_info_20260527_145700_strict.json}"

echo "PYTHON_BIN=$PYTHON_BIN"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CLASS_LIST=$CLASS_LIST"
echo "SAMPLES_PER_CLASS=$SAMPLES_PER_CLASS"
echo "NEGATIVE_COUNT=$NEGATIVE_COUNT"
echo "RANDOM_SEED=$RANDOM_SEED"
echo "SPLIT_INFO=$SPLIT_INFO"

mkdir -p "$OUTPUT_DIR"
cp "$CLASS_LIST" "$OUTPUT_DIR/hard_classes.txt"

"$PYTHON_BIN" tools/vloom_agrinet/prepare_disease_pest_data.py \
  --data-root datasets/AgriNet-1K \
  --feature-root outputs/vit-base-cluster/vit-base-1 \
  --output-dir "$OUTPUT_DIR" \
  --samples-per-class "$SAMPLES_PER_CLASS" \
  --negative-count "$NEGATIVE_COUNT" \
  --reuse-pairs outputs/vlm_data/disease_pest/finegrained_pairs_vit_base.jsonl \
  --train-split-info "$SPLIT_INFO" \
  --class-list "$CLASS_LIST" \
  --random-seed "$RANDOM_SEED"

"$PYTHON_BIN" tools/vloom_agrinet/validate_disease_pest_data.py \
  --data-dir "$OUTPUT_DIR" \
  --expected-disease 29 \
  --expected-pest 16 \
  --top-k 5 \
  --expected-negatives "$NEGATIVE_COUNT" \
  --samples-per-class "$SAMPLES_PER_CLASS"

"$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

sample_path = Path(sys.argv[1]) / "contrast_samples_vit_base.jsonl"
rows = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line.strip()]
representative_negatives = sum(
    1
    for row in rows
    for neg in row.get("negative_reference_images", [])
    if neg.get("filled_from_representative")
)
if representative_negatives:
    raise SystemExit(f"representative negatives remain: {representative_negatives}")
print(f"samples={len(rows)}")
print(f"classes={len(Counter(row['final_label'] for row in rows))}")
print("representative_negatives=0")
PY

echo "Prepared hard random10 samples under $OUTPUT_DIR"
