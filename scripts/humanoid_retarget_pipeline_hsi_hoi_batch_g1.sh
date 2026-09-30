#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${REPO_ROOT}/sample_data/omomo"
OUTPUT_DIR="${REPO_ROOT}/output/unitree_g1_retarget"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if (( $# > 0 )) && [[ "$1" != -* ]]; then
    INPUT_DIR="$1"
    shift
fi
if (( $# > 0 )) && [[ "$1" != -* ]]; then
    OUTPUT_DIR="$1"
    shift
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/humanoid_retarget_pipeline_hsi_hoi_batch.py" \
    --config "${REPO_ROOT}/robot_configs/humanoid_retarget_unitree_g1_example.json" \
    --defaults "${REPO_ROOT}/humanoid_retarget_defaults_hsi_hoi_standard.json" \
    --data "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    "$@"
