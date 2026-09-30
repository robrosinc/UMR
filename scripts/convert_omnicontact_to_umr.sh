#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${1:-/home/robros/workspace/motion_datas/OmniContact}"
OUTPUT_DIR="${2:-${REPO_ROOT}/sample_data/omnicontact}"
PYTHON_BIN="${PYTHON_BIN:-/home/robros/workspace/miniconda3/envs/UMR/bin/python}"

if (( $# >= 2 )); then
    shift 2
elif (( $# == 1 )); then
    shift
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/convert_omnicontact_to_umr.py" \
    --input "${INPUT_DIR}" --output "${OUTPUT_DIR}" "$@"
