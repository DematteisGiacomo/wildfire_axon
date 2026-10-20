#!/usr/bin/env bash
# Create the Python 3.11 virtual environment used for training and Axon compilation.
#
# Usage: ./setup_venv.sh [path-to-sdk-edge-ai]
set -euo pipefail
cd "$(dirname "$0")"

SDK_EDGE_AI="${1:-../edge-ai}"
VENV_DIR=".venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating Python 3.11 virtual environment in ${VENV_DIR}/"
  if command -v uv >/dev/null; then
    uv venv --python 3.11 "$VENV_DIR"
  else
    python3.11 -m venv "$VENV_DIR"
  fi
fi

if command -v uv >/dev/null; then
  PIP=(uv pip install --python "${VENV_DIR}/bin/python")
else
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
  PIP=("${VENV_DIR}/bin/python" -m pip install)
fi
"${PIP[@]}" -r requirements.txt -r "${SDK_EDGE_AI}/tools/axon/compiler/scripts/requirements.txt"

echo "Virtual environment ready. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
