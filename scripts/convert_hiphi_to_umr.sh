#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${REPO_ROOT}/../motion_datas/HiPHI_origin"
OUTPUT_DIR="${REPO_ROOT}/sample_data/hiphi"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${REPO_ROOT}/external_assets/hiphi/SMPLX_NEUTRAL.npz"

if (( $# > 0 )) && [[ "$1" != -* ]]; then
    INPUT_DIR="$1"
    shift
fi
if (( $# > 0 )) && [[ "$1" != -* ]]; then
    OUTPUT_DIR="$1"
    shift
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/convert_hiphi_to_umr.py" \
    --input "${INPUT_DIR}" --output "${OUTPUT_DIR}" \
    --model-path "${MODEL_PATH}" \
    --workers 4 --beta-iters 1 --mink-iters 1 "$@"

# default
# --beta-iters 100 --mink-iters 10
