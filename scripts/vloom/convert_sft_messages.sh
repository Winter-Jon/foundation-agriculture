#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

INPUT_JSON="${INPUT_JSON:-outputs/vloom/contrast_cot/agrinet_disease_pest_contrast_cot_real/agrinet_disease_pest/agrinet_contrast_cot_results.json}"
OUTPUT_JSONL="${OUTPUT_JSONL:-outputs/vlm_data/disease_pest/sft_messages.jsonl}"

exec "$PYTHON_BIN" src/agrinet/data/vloom_tools/convert_to_sft_messages.py \
  --input "$INPUT_JSON" \
  --output "$OUTPUT_JSONL" \
  "$@"
