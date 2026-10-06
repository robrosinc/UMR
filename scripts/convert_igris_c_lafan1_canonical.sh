#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON_BIN:-/home/robros/workspace/miniconda3/envs/UMR/bin/python}" \
    "${ROOT}/scripts/convert_igris_c_lafan1_canonical.py" \
    --input "${ROOT}/output/igris_c_lafan1/igris_c" \
    --output-root "${ROOT}/output/igris_c_lafan1_canonical" \
    "$@"
