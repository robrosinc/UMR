#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="/home/robros/workspace/motion_datas/omomo"
OUTPUT_DIR="${REPO_ROOT}/sample_data/omomo"
# The joblib files have no FPS metadata. The released sequences are 30 FPS;
# override FPS if using a differently sampled copy.
FPS="${FPS:-30}"
COLLISION="${COLLISION:-coacd}"
COACD_THRESHOLD="${COACD_THRESHOLD:-0.03}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if (( $# > 0 )) && [[ "$1" != -* ]]; then
    INPUT_DIR="$1"
    shift
fi
if (( $# > 0 )) && [[ "$1" != -* ]]; then
    OUTPUT_DIR="$1"
    shift
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/convert_omomo_to_umr.py" \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    --fps "${FPS}" \
    --collision "${COLLISION}" \
    --coacd-threshold "${COACD_THRESHOLD}" \
    "$@"
