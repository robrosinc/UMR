#!/usr/bin/env bash
set -euo pipefail

# Edit these paths or set the matching environment variables. Extra Python options may follow.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${INPUT_DIR:-$ROOT/../GMR/pico/retargeted}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/light_form/igris_c_pico}"
ROBOT_XML="${ROBOT_XML:-$ROOT/../GMR/assets/robros_igris_c_v2/igris_c_v2.xml}"
PYTHON_BIN="${UMR_PYTHON:-/home/robros/workspace/miniconda3/envs/UMR/bin/python}"

"$PYTHON_BIN" "$ROOT/scripts/convert_pico_pkl_to_light.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --robot-xml "$ROBOT_XML" \
  "$@"
