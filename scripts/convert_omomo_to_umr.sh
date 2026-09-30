#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${1:-/home/robros/workspace/motion_datas/omomo}"
OUTPUT_DIR="${2:-${REPO_ROOT}/sample_data/omomo}"
# The joblib files have no FPS metadata. The released sequences are 30 FPS;
# override FPS if using a differently sampled copy.
FPS="${FPS:-30}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if (( $# >= 2 )); then
    shift 2
elif (( $# == 1 )); then
    shift
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/convert_omomo_to_umr.py" \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    --fps "${FPS}" \
    "$@"
