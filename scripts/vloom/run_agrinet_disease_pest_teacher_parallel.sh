#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${YUNWU_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Set YUNWU_API_KEY or OPENAI_API_KEY before running teacher collection." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

CONFIG_PATH="${CONFIG_PATH:-configs/vloom/agrinet_disease_pest_contrast_cot_real.yaml}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"

exec "$PYTHON_BIN" -m tools.vloom_agrinet.run_contrast_cot \
  --config_path "$CONFIG_PATH" \
  --max-concurrent "$MAX_CONCURRENT" \
  "$@"
