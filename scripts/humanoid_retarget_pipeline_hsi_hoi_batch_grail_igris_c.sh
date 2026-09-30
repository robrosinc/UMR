#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${REPO_ROOT}/sample_data/grail"
OUTPUT_DIR="${REPO_ROOT}/output/igris_c_grail_retarget"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        PYTHON_BIN="/home/robros/workspace/miniconda3/envs/UMR/bin/python"
    fi
fi

if (( $# > 0 )) && [[ "$1" != -* ]]; then
    INPUT_DIR="$1"
    shift
fi
if (( $# > 0 )) && [[ "$1" != -* ]]; then
    OUTPUT_DIR="$1"
    shift
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/humanoid_retarget_pipeline_hsi_hoi_batch.py" \
    --config "${REPO_ROOT}/robot_configs/humanoid_retarget_igris_c_grail_example.json" \
    --defaults "${REPO_ROOT}/humanoid_retarget_defaults_hsi_hoi_grail.json" \
    --data "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    "$@"
