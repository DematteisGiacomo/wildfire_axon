#!/usr/bin/env bash
# Compile the int8 TFLite model for the Axon NPU and install the header in the
# image_classification application.
#
# Usage: ./compile_axon.sh [path-to-sdk-edge-ai] [path-to-image_classification-app]
set -euo pipefail
cd "$(dirname "$0")"

SDK_EDGE_AI="$(realpath "${1:-../edge-ai}")"
APP_DIR="$(realpath "${2:-${SDK_EDGE_AI}/applications/image_classification}")"
SCRIPTS="${SDK_EDGE_AI}/tools/axon/compiler/scripts"
PYTHON=".venv/bin/python"

"$PYTHON" prepare_compiler_data.py
"$PYTHON" "${SCRIPTS}/axons_tflite_model_scan.py" trained_models/wildfire_classifier_int8.tflite
"$PYTHON" "${SCRIPTS}/axons_ml_nn_compiler_executor.py" compiler_input_yaml_wildfire.yaml

install -m 644 outputs/nrf_axon_model_wildfire_.h "${APP_DIR}/src/generated/"
echo "Installed outputs/nrf_axon_model_wildfire_.h in ${APP_DIR}/src/generated/"
grep -E "MAX_IL_BUFFER_USED|PACKED_OUTPUT_SIZE" outputs/nrf_axon_model_wildfire_.h | head -n 4
