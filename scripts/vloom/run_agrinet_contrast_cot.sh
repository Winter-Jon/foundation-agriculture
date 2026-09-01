#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VLOOM_BIN="${VLOOM_BIN:-.venv/bin/vloom}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

CONFIG_PATH="${CONFIG_PATH:-configs/vloom/agrinet_insect_contrast_cot.yaml}"

exec "$PYTHON_BIN" -m agrinet.data.vloom_tools.run_contrast_cot --config_path "$CONFIG_PATH" "$@"
