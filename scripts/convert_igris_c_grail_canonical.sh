#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/robros/workspace/UMR
PYTHON_BIN=${UMR_PYTHON:-/home/robros/workspace/miniconda3/envs/UMR/bin/python}

"$PYTHON_BIN" "$ROOT/scripts/convert_igris_c_grail_canonical.py" \
    --input "$ROOT/output/igris_c_grail" \
    --output-root "$ROOT/output/igris_c_grail_canonical" \
    "$@"
