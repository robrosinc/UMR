#!/usr/bin/env bash
set -euo pipefail

# Override these paths in the environment or pass Python CLI options.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${INPUT_DIR:-$ROOT/output/igris_c_grail}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/light_form/igris_c_grail}"
SAMPLE_ROOT="${SAMPLE_ROOT:-$ROOT/sample_data/grail}"
ROBOT_XML="${ROBOT_XML:-$ROOT/assets/igris_c/igris_c.xml}"
PYTHON_BIN="${UMR_PYTHON:-/home/robros/workspace/miniconda3/envs/UMR/bin/python}"

"$PYTHON_BIN" "$ROOT/scripts/convert_grail_umr_to_light.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --sample-root "$SAMPLE_ROOT" \
  --robot-xml "$ROBOT_XML" \
  "$@"
