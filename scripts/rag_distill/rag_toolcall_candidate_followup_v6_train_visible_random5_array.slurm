#!/bin/bash
#SBATCH --job-name=rag-v6-trainvis-r5
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-139%4
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="/home/jiangwentao/Repos/foundation-agriculture"
cd "$REPO_ROOT"

source /etc/profile.d/modules.sh
module purge
module load cuda/12.8

export HF_HOME="${HF_HOME:-/home/jiangwentao/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export http_proxy="${http_proxy:-http://10.10.10.10:3128}"
export https_proxy="${https_proxy:-http://10.10.10.10:3128}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
RAG_HOST="${RAG_HOST:-127.0.0.1}"
BASE_RAG_PORT="${BASE_RAG_PORT:-8200}"
RAG_PORT="$((BASE_RAG_PORT + SLURM_ARRAY_TASK_ID))"
RAG_API="http://${RAG_HOST}:${RAG_PORT}"
MILVUS_LITE_DB="${MILVUS_LITE_DB:-outputs/milvus/agrinet_wiki_lite.db}"
TASK_TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/rag_v6_trainvis_r5_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
TASK_MILVUS_LITE_DB="$TASK_TMP_ROOT/agrinet_wiki_lite.db"
TASK_MILVUS_HOME="$TASK_TMP_ROOT/milvus_home"
COLLECTION="${COLLECTION:-agrinet_wiki_siglip2}"
MODEL_NAME="${MODEL_NAME:-models/siglip2-so400m-patch16-naflex}"
SAMPLE_FILE="${SAMPLE_FILE:-outputs/vlm_data/disease_pest_train_visible_random5/contrast_samples_vit_base.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rag_distill/agrinet_rag_toolcall_v6_train_visible_random5}"
SHARD_SIZE="${SHARD_SIZE:-5}"
OFFSET="$((SLURM_ARRAY_TASK_ID * SHARD_SIZE))"
OUTPUT_DIR="${OUTPUT_ROOT}/shard_${SLURM_ARRAY_TASK_ID}"
TEACHER_MODEL="${TEACHER_MODEL:-gpt-5.4-mini}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-5}"

RAG_PID=""
cleanup() {
  if [[ -n "$RAG_PID" ]] && kill -0 "$RAG_PID" 2>/dev/null; then
    echo "=== Stop RAG API pid=$RAG_PID ==="
    kill "$RAG_PID" || true
    wait "$RAG_PID" || true
  fi
}
trap cleanup EXIT

if [[ -z "${YUNWU_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" && -x "$HOME/.apikeys/bin/apikey" ]]; then
  eval "$($HOME/.apikeys/bin/apikey env yunwu)"
fi

if [[ -z "${YUNWU_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "YUNWU_API_KEY or OPENAI_API_KEY is required. Export it before sbatch and use --export=ALL." >&2
  exit 2
fi

echo "HOSTNAME=$HOSTNAME"
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "RAG_API=$RAG_API"
echo "MILVUS_LITE_DB=$MILVUS_LITE_DB"
echo "TASK_MILVUS_LITE_DB=$TASK_MILVUS_LITE_DB"
echo "TASK_MILVUS_HOME=$TASK_MILVUS_HOME"
echo "SAMPLE_FILE=$SAMPLE_FILE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "SHARD_SIZE=$SHARD_SIZE"
echo "OFFSET=$OFFSET"
echo "TEACHER_MODEL=$TEACHER_MODEL"
echo "REASONING_EFFORT=$REASONING_EFFORT"
echo "MAX_TOOL_TURNS=$MAX_TOOL_TURNS"

echo "=== Start RAG API ==="
mkdir -p "$TASK_TMP_ROOT"
mkdir -p "$TASK_MILVUS_HOME"
cp "$MILVUS_LITE_DB" "$TASK_MILVUS_LITE_DB"
export MILVUS_LITE_CACHE="$TASK_MILVUS_HOME"
export MILVUS_LITE_DATA_PATH="$TASK_MILVUS_HOME"
export XDG_CACHE_HOME="$TASK_TMP_ROOT/xdg_cache"
"$PYTHON_BIN" tools/milvus/search_api.py \
  --host "$RAG_HOST" \
  --port "$RAG_PORT" \
  --device cpu \
  --mode lite \
  --lite-db "$TASK_MILVUS_LITE_DB" \
  --collection "$COLLECTION" \
  --model-name "$MODEL_NAME" &
RAG_PID=$!

for attempt in $(seq 1 60); do
  if curl -sS "$RAG_API/health" >/dev/null; then
    echo "=== RAG API health ok ==="
    break
  fi
  if ! kill -0 "$RAG_PID" 2>/dev/null; then
    echo "RAG API exited before health check succeeded" >&2
    wait "$RAG_PID"
  fi
  if [[ "$attempt" -eq 60 ]]; then
    echo "Timed out waiting for RAG API health" >&2
    exit 1
  fi
  sleep 10
done

echo "=== Start train-visible random5 distillation shard ==="
"$PYTHON_BIN" -m tools.rag_distill.run_pilot \
  --sample-file "$SAMPLE_FILE" \
  --offset "$OFFSET" \
  --limit "$SHARD_SIZE" \
  --rag-api "$RAG_API" \
  --output-dir "$OUTPUT_DIR" \
  --model "$TEACHER_MODEL" \
  --max-tool-turns "$MAX_TOOL_TURNS" \
  --top-k 3 \
  --max-concurrent 1 \
  --reasoning-effort "$REASONING_EFFORT"

echo "=== Validate artifact ==="
"$PYTHON_BIN" -m tools.rag_distill.validate_artifact \
  --artifact-dir "$OUTPUT_DIR"

echo "=== RAG train-visible random5 shard complete ==="
