#!/usr/bin/env bash
set -euo pipefail

# Edit these paths, then run this script. Extra Python options may follow.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="$ROOT/output/igris_c_bones_seed/igris_c"
OUTPUT_DIR="$ROOT/light_form/igris_c_bones_seed"
ROBOT_XML="$ROOT/assets/igris_c/igris_c.xml"
PYTHON_BIN="${UMR_PYTHON:-/home/robros/workspace/miniconda3/envs/UMR/bin/python}"

"$PYTHON_BIN" "$ROOT/scripts/convert_bones_seed_umr_to_light.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --robot-xml "$ROBOT_XML" \
  "$@"
