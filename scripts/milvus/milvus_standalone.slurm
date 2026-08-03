#!/bin/bash
#SBATCH --job-name=milvus-standalone
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="/home/jiangwentao/Repos/foundation-agriculture"
cd "$REPO_ROOT"

source /etc/profile.d/modules.sh
module purge
module load cuda/12.8

MILVUS_IMAGE="${MILVUS_IMAGE:-milvusdb/milvus:v2.5.4}"
MILVUS_CONTAINER="${MILVUS_CONTAINER:-agrinet-milvus-${SLURM_JOB_ID}}"
MILVUS_PORT="${MILVUS_PORT:-19530}"
MILVUS_WEB_PORT="${MILVUS_WEB_PORT:-9091}"
MILVUS_DATA_DIR="${MILVUS_DATA_DIR:-$REPO_ROOT/outputs/milvus/standalone}"

mkdir -p "$MILVUS_DATA_DIR" "$REPO_ROOT/slurm"

echo "HOSTNAME=$HOSTNAME"
echo "MILVUS_IMAGE=$MILVUS_IMAGE"
echo "MILVUS_CONTAINER=$MILVUS_CONTAINER"
echo "MILVUS_URI=http://$HOSTNAME:$MILVUS_PORT"
echo "MILVUS_DATA_DIR=$MILVUS_DATA_DIR"

cleanup() {
    docker rm -f "$MILVUS_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker rm -f "$MILVUS_CONTAINER" >/dev/null 2>&1 || true
docker run --name "$MILVUS_CONTAINER" \
    -p "${MILVUS_PORT}:19530" \
    -p "${MILVUS_WEB_PORT}:9091" \
    -v "$MILVUS_DATA_DIR:/var/lib/milvus" \
    "$MILVUS_IMAGE" \
    milvus run standalone &

CONTAINER_LAUNCH_PID=$!
echo "docker_launch_pid=$CONTAINER_LAUNCH_PID"

"$REPO_ROOT/.venv/bin/python" tools/milvus/check_milvus_ready.py \
    --uri "http://127.0.0.1:${MILVUS_PORT}" \
    --timeout 600 \
    --interval 10

echo "milvus_ready host=$HOSTNAME port=$MILVUS_PORT"
echo "Keep this Slurm job running while ingest/query jobs connect to http://$HOSTNAME:$MILVUS_PORT"

while docker ps --filter "name=${MILVUS_CONTAINER}" --format '{{.Names}}' | grep -qx "$MILVUS_CONTAINER"; do
    sleep 60
done

echo "Milvus container exited"
