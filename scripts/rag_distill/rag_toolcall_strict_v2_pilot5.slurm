#!/bin/bash
#SBATCH --job-name=rag-toolcall-v3-pilot5
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

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
RAG_PORT="${RAG_PORT:-8077}"
RAG_API="http://${RAG_HOST}:${RAG_PORT}"
MILVUS_LITE_DB="${MILVUS_LITE_DB:-outputs/milvus/agrinet_wiki_lite.db}"
COLLECTION="${COLLECTION:-agrinet_wiki_siglip2}"
MODEL_NAME="${MODEL_NAME:-models/siglip2-so400m-patch16-naflex}"
SAMPLE_FILE="${SAMPLE_FILE:-outputs/vlm_data/disease_pest_test/contrast_samples_vit_base.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rag_distill/agrinet_rag_toolcall_v3_name_top3_pilot5}"
TEACHER_MODEL="${TEACHER_MODEL:-gpt-5.4-mini}"

RAG_PID=""
cleanup() {
  if [[ -n "$RAG_PID" ]] && kill -0 "$RAG_PID" 2>/dev/null; then
    echo "=== Stop RAG API pid=$RAG_PID ==="
    kill "$RAG_PID" || true
    wait "$RAG_PID" || true
  fi
}
trap cleanup EXIT

if [[ -z "${YUNWU_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "YUNWU_API_KEY or OPENAI_API_KEY is required. Export it in the submit shell and use sbatch --export=ALL." >&2
  exit 2
fi

echo "HOSTNAME=$HOSTNAME"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "RAG_API=$RAG_API"
echo "MILVUS_LITE_DB=$MILVUS_LITE_DB"
echo "COLLECTION=$COLLECTION"
echo "MODEL_NAME=$MODEL_NAME"
echo "SAMPLE_FILE=$SAMPLE_FILE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "TEACHER_MODEL=$TEACHER_MODEL"

echo "=== Start RAG API ==="
"$PYTHON_BIN" tools/milvus/search_api.py \
  --host "$RAG_HOST" \
  --port "$RAG_PORT" \
  --device cpu \
  --mode lite \
  --lite-db "$MILVUS_LITE_DB" \
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

echo "=== Start strict v2 teacher distillation ==="
"$PYTHON_BIN" -m tools.rag_distill.run_pilot \
  --sample-file "$SAMPLE_FILE" \
  --limit 5 \
  --rag-api "$RAG_API" \
  --output-dir "$OUTPUT_DIR" \
  --model "$TEACHER_MODEL" \
  --max-tool-turns 3 \
  --top-k 3 \
  --max-concurrent 1

echo "=== Convert raw teacher traces to student SFT ==="
"$PYTHON_BIN" -m tools.rag_distill.convert_to_student_sft \
  --artifact-dir "$OUTPUT_DIR"

echo "=== Validate student artifact ==="
"$PYTHON_BIN" -m tools.rag_distill.validate_artifact \
  --artifact-dir "$OUTPUT_DIR" \
  --train-file "$OUTPUT_DIR/train/agent_sft.student.jsonl"

echo "=== RAG strict v2 tool-call pilot complete ==="
