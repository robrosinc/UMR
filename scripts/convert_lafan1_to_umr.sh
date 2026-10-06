#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/robros/workspace/miniconda3/envs/UMR/bin/python}"
INPUT="/home/robros/workspace/motion_datas/lafan1"
OUTPUT="${REPO_ROOT}/sample_data/lafan1_smplx"

if (( $# > 0 )) && [[ "$1" != -* ]]; then
    INPUT="$1"
    shift
fi
if (( $# > 0 )) && [[ "$1" != -* ]]; then
    OUTPUT="$1"
    shift
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/convert_lafan1_to_umr.py" \
    --input "${INPUT}" --output "${OUTPUT}" "$@"
