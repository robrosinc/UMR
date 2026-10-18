#!/usr/bin/env bash
set -euo pipefail

# Edit these paths, then run this script. Extra Python options may follow.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="$ROOT/output/igris_c_amass/igris_c"
OUTPUT_DIR="$ROOT/light_form/igris_c_amass"
PYTHON_BIN="${UMR_PYTHON:-/home/robros/workspace/miniconda3/envs/UMR/bin/python}"

"$PYTHON_BIN" "$ROOT/scripts/convert_amass_umr_to_light.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
