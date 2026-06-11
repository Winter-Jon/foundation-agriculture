#!/bin/bash
#SBATCH --job-name=agrinet-distill-large
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/agrinet-distill-large_%j.out
#SBATCH --error=slurm/agrinet-distill-large_%j.err

set -euo pipefail

REPO_ROOT="/home/jiangwentao/Repos/foundation-agriculture"
cd "$REPO_ROOT"

if [[ -z "${YUNWU_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Set YUNWU_API_KEY or OPENAI_API_KEY before running teacher collection." >&2
  exit 2
fi

# Use Python 3.12 venv (created with uv for compute-node compatibility)
PYTHON_BIN="$REPO_ROOT/.venv-py312/bin/python"

# Ensure VLooM source is on PYTHONPATH
export PYTHONPATH="${REPO_ROOT}:/home/jiangwentao/Repos/VLooM/src:${PYTHONPATH:-}"

# Set proxy for internet access on compute nodes
export http_proxy="http://10.10.10.10:3128"
export https_proxy="http://10.10.10.10:3128"

# ------------------------------------------------------------------
# Stage 1: Prepare data (10 samples per class ~ 2.1k total)
# ------------------------------------------------------------------
echo "=== Stage 1: Prepare large batch data ==="
"$PYTHON_BIN" tools/vloom_agrinet/prepare_disease_pest_data.py \
  --data-root datasets/AgriNet-1K \
  --feature-root outputs/vit-base-cluster/vit-base-1 \
  --output-dir outputs/vlm_data/disease_pest_large \
  --samples-per-class 5 \
  --reuse-pairs outputs/vlm_data/disease_pest/finegrained_pairs_vit_base.jsonl

SAMPLES=$(wc -l < outputs/vlm_data/disease_pest_large/contrast_samples_vit_base.jsonl)
echo "Prepared $SAMPLES contrast samples"

# ------------------------------------------------------------------
# Stage 2: English teacher collection
# ------------------------------------------------------------------
echo "=== Stage 2: English teacher collection ==="
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
"$PYTHON_BIN" -m tools.vloom_agrinet.run_contrast_cot \
  --config_path configs/vloom/agrinet_disease_pest_large_batch.yaml \
  --max-concurrent "$MAX_CONCURRENT"

RESULT_JSON_EN="outputs/vloom/contrast_cot/agrinet_disease_pest_large_batch/agrinet_disease_pest/agrinet_contrast_cot_results.json"
if [[ ! -f "$RESULT_JSON_EN" ]]; then
  echo "English teacher results not found: $RESULT_JSON_EN" >&2
  exit 1
fi

# ------------------------------------------------------------------
# Stage 3: Chinese teacher collection
# ------------------------------------------------------------------
echo "=== Stage 3: Chinese teacher collection ==="
"$PYTHON_BIN" -m tools.vloom_agrinet.run_contrast_cot \
  --config_path configs/vloom/agrinet_disease_pest_large_batch_zh.yaml \
  --max-concurrent "$MAX_CONCURRENT"

RESULT_JSON_ZH="outputs/vloom/contrast_cot/agrinet_disease_pest_large_batch_zh/agrinet_disease_pest/agrinet_contrast_cot_zh_results.json"
if [[ ! -f "$RESULT_JSON_ZH" ]]; then
  echo "Chinese teacher results not found: $RESULT_JSON_ZH" >&2
  exit 1
fi

# ------------------------------------------------------------------
# Stage 4: Convert English results to English SFT
# ------------------------------------------------------------------
echo "=== Stage 4: Convert English teacher results ==="
"$PYTHON_BIN" tools/vloom_agrinet/convert_to_sft_messages.py \
  --input "$RESULT_JSON_EN" \
  --output-sft-en outputs/vlm_data/disease_pest_large/sft_messages_en.jsonl \
  --output-sft-zh outputs/vlm_data/disease_pest_large/sft_messages_zh_from_en_temp.jsonl \
  --output-traces outputs/vlm_data/disease_pest_large/teacher_traces_en.jsonl \
  --bilingual-wiki outputs/vlm_data/disease_pest/wiki_bilingual_structured.json

rm -f outputs/vlm_data/disease_pest_large/sft_messages_zh_from_en_temp.jsonl

# ------------------------------------------------------------------
# Stage 5: Convert Chinese results to Chinese SFT
# ------------------------------------------------------------------
echo "=== Stage 5: Convert Chinese teacher results ==="
"$PYTHON_BIN" tools/vloom_agrinet/convert_to_sft_messages.py \
  --input "$RESULT_JSON_ZH" \
  --output-sft-en outputs/vlm_data/disease_pest_large/sft_messages_en_from_zh_temp.jsonl \
  --output-sft-zh outputs/vlm_data/disease_pest_large/sft_messages_zh.jsonl \
  --output-traces outputs/vlm_data/disease_pest_large/teacher_traces_zh.jsonl \
  --bilingual-wiki outputs/vlm_data/disease_pest/wiki_bilingual_structured.json

rm -f outputs/vlm_data/disease_pest_large/sft_messages_en_from_zh_temp.jsonl

echo "=== Large batch distillation complete ==="
echo "Outputs under outputs/vlm_data/disease_pest_large/"
echo "  - sft_messages_en.jsonl  (from English teacher)"
echo "  - sft_messages_zh.jsonl  (from Chinese teacher)"
echo "  - teacher_traces_en.jsonl"
echo "  - teacher_traces_zh.jsonl"
