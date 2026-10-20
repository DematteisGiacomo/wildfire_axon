#!/usr/bin/env bash
# End-to-end pipeline: dataset, training, int8 quantization, evaluation, Axon compilation.
#
# Usage: ./train_all.sh [path-to-sdk-edge-ai]
set -euo pipefail
cd "$(dirname "$0")"

SDK_EDGE_AI="${1:-../edge-ai}"
PYTHON=".venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Virtual environment not found. Running setup_venv.sh ..."
  ./setup_venv.sh "$SDK_EDGE_AI"
fi

echo "=== 1. Download and cache dataset ==="
"$PYTHON" prepare_dataset.py

echo "=== 2. Train model ==="
"$PYTHON" train.py

echo "=== 3. Quantize to TFLite ==="
"$PYTHON" quantize.py

echo "=== 4. Evaluate ==="
"$PYTHON" evaluate.py

echo "=== 5. Compile for Axon ==="
./compile_axon.sh "$SDK_EDGE_AI"
