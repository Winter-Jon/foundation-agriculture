#!/bin/bash
#SBATCH --job-name=qwen3vl-eval-sft
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="/home/jiangwentao/Repos/foundation-agriculture"
cd "$REPO_ROOT"

source /etc/profile.d/modules.sh
module purge
module load cuda/12.8

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${SLURM_JOB_GPUS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
fi

export PYTHONPATH="$REPO_ROOT/vlm/eval/VLMEvalKit:$REPO_ROOT/vlm/sft/ms-swift:$REPO_ROOT:${PYTHONPATH:-}"
export LMUData="$REPO_ROOT/tmp/LMUData"
export SPLIT_THINK=True
export PRED_FORMAT=tsv
export EVAL_FORMAT=json

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
SWIFT_CMD=("$PYTHON_BIN" -m swift.cli.main)
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-0}"
RUN_NAME="${RUN_NAME:-sft_large_${SPLIT}}"
SFT_ROOT="${SFT_ROOT:-outputs/vlm_sft/qwen3_vl_4b_disease_pest_large}"
OUT_DIR="outputs/vlm_eval/qwen3_vl_4b_disease_pest/${RUN_NAME}"

mkdir -p "$OUT_DIR" "$LMUData"

if [[ -z "${MODEL_PATH:-}" ]]; then
    LATEST_CHECKPOINT="$(find "$SFT_ROOT" -mindepth 2 -maxdepth 2 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
    if [[ -z "$LATEST_CHECKPOINT" || ! -d "$LATEST_CHECKPOINT" ]]; then
        echo "No SFT checkpoint found under $SFT_ROOT/*/checkpoint-*" >&2
        exit 2
    fi
    if [[ -f "$LATEST_CHECKPOINT/adapter_config.json" ]]; then
        MODEL_PATH="${LATEST_CHECKPOINT%/}-merged"
        if [[ ! -f "$MODEL_PATH/config.json" ]]; then
            echo "Merging LoRA adapter: $LATEST_CHECKPOINT -> $MODEL_PATH"
            "${SWIFT_CMD[@]}" export \
                --adapters "$LATEST_CHECKPOINT" \
                --output_dir "$MODEL_PATH" \
                --merge_lora true
        fi
    else
        MODEL_PATH="$LATEST_CHECKPOINT"
    fi
fi

if [[ -f "$MODEL_PATH/adapter_config.json" ]]; then
    MERGED_MODEL_PATH="${MODEL_PATH%/}-merged"
    if [[ ! -f "$MERGED_MODEL_PATH/config.json" ]]; then
        echo "Merging explicit LoRA MODEL_PATH: $MODEL_PATH -> $MERGED_MODEL_PATH"
        "${SWIFT_CMD[@]}" export \
            --adapters "$MODEL_PATH" \
            --output_dir "$MERGED_MODEL_PATH" \
            --merge_lora true
    fi
    MODEL_PATH="$MERGED_MODEL_PATH"
fi

if [[ ! -e "$MODEL_PATH" ]]; then
    echo "MODEL_PATH does not exist: $MODEL_PATH" >&2
    exit 2
fi

if [[ "$SPLIT" == "large" ]]; then
    TRACE_EN="outputs/vlm_data/disease_pest_large/teacher_traces_en.jsonl"
    TRACE_ZH="outputs/vlm_data/disease_pest_large/teacher_traces_zh.jsonl"
else
    TRACE_EN="outputs/vlm_data/disease_pest_test/teacher_traces_en.jsonl"
    TRACE_ZH="outputs/vlm_data/disease_pest_test/teacher_traces_zh.jsonl"
fi

if [[ ! -f "$TRACE_EN" || ! -f "$TRACE_ZH" ]]; then
    echo "Missing eval traces: $TRACE_EN or $TRACE_ZH" >&2
    exit 2
fi

MANIFEST="$OUT_DIR/manifest.jsonl"
DATA_TSV="$OUT_DIR/agrinet_vlmeval.tsv"
CONFIG="$OUT_DIR/vlmeval_config.json"
SCORED="$OUT_DIR/scored.jsonl"
METRICS="$OUT_DIR/metrics.json"
CSV="$OUT_DIR/scored.csv"

BUILD_ARGS=(--traces "$TRACE_EN" "$TRACE_ZH" --output "$MANIFEST" --output-tsv "$DATA_TSV" --repo-root "$REPO_ROOT" --data-dir "$(dirname "$TRACE_EN")")
if [[ "$LIMIT" != "0" ]]; then
    BUILD_ARGS+=(--limit "$LIMIT")
fi
"$PYTHON_BIN" vlm/eval/tools/build_eval_manifest.py "${BUILD_ARGS[@]}"

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

config = {
    "model": {
        "Qwen3VL-4B-AgriNet-SFT": {
            "class": "Qwen3VLLocalChat",
            "model_path": "$MODEL_PATH",
            "max_new_tokens": int("${MAX_NEW_TOKENS:-128}"),
        }
    },
    "data": {
        "AgriNetVLM": {
            "class": "AgriNetVLM",
            "dataset": "AgriNetVLM",
            "data_file": "$DATA_TSV",
            "repo_root": "$REPO_ROOT",
        }
    },
}
Path("$CONFIG").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

"$PYTHON_BIN" vlm/eval/VLMEvalKit/run.py \
    --config "$CONFIG" \
    --work-dir "$OUT_DIR" \
    --reuse

PREDICTIONS="$(find "$OUT_DIR/Qwen3VL-4B-AgriNet-SFT" -name 'Qwen3VL-4B-AgriNet-SFT_AgriNetVLM.tsv' -type f | sort | tail -n 1)"
if [[ -z "$PREDICTIONS" || ! -f "$PREDICTIONS" ]]; then
    echo "Missing VLMEvalKit prediction TSV under $OUT_DIR/Qwen3VL-4B-AgriNet-SFT" >&2
    exit 3
fi
cp "$PREDICTIONS" "$OUT_DIR/predictions.tsv"

"$PYTHON_BIN" vlm/eval/tools/normalize_answers.py \
    --predictions "$PREDICTIONS" \
    --output-jsonl "$SCORED" \
    --output-metrics "$METRICS" \
    --output-csv "$CSV"
