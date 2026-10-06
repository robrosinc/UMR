#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

exec "${PYTHON_BIN:-/home/robros/workspace/miniconda3/envs/UMR/bin/python}" \
    scripts/humanoid_retarget_pipeline_batch.py \
    --config robot_configs/humanoid_retarget_igris_c_example.json \
    --batch-config humanoid_retarget_defaults_batch.json \
    --pattern '*.npz' \
    --output-root output/igris_c_lafan1 \
    --workers "${WORKERS:-8}" \
    "$@"
