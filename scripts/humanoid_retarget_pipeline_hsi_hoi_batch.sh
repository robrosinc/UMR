#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${1:-${REPO_ROOT}/sample_data/omomo}"
OUTPUT_DIR="${2:-${REPO_ROOT}/output/igris_c_retarget}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if (( $# >= 2 )); then
    shift 2
elif (( $# == 1 )); then
    shift
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/humanoid_retarget_pipeline_hsi_hoi_batch.py" \
    --config "${REPO_ROOT}/robot_configs/humanoid_retarget_igris_c_example.json" \
    --defaults "${REPO_ROOT}/humanoid_retarget_defaults_hsi_hoi_standard.json" \
    --data "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    "$@"
