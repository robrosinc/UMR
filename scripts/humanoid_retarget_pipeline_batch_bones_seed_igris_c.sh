#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${REPO_ROOT}/sample_data/bones-seed/motions_uniform/bvh"
OUTPUT_DIR="${REPO_ROOT}/output/bones_seed_igris_c_retarget"
PYTHON_BIN="${PYTHON_BIN:-/home/robros/workspace/miniconda3/envs/UMR/bin/python}"

if (( $# > 0 )) && [[ "$1" != -* ]]; then
    INPUT_DIR="$1"
    shift
fi
if (( $# > 0 )) && [[ "$1" != -* ]]; then
    OUTPUT_DIR="$1"
    shift
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/humanoid_retarget_pipeline_batch_bones_seed_igris_c.py" \
    --motion-folder "${INPUT_DIR}" --output-root "${OUTPUT_DIR}" "$@"
