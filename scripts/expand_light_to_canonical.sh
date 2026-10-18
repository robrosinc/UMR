#!/usr/bin/env bash
set -euo pipefail

# Override these environment variables or pass --input-dir/--output-dir.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${LIGHT_INPUT_DIR:-$ROOT/light_form/igris_c_hiphi}"
OUTPUT_DIR="${CANONICAL_OUTPUT_DIR:-$ROOT/canonical_form/igris_c_hiphi}"
PYTHON_BIN="${LIGHT_EXPAND_PYTHON:-${UMR_PYTHON:-python3}}"

"$PYTHON_BIN" "$ROOT/scripts/expand_light_to_canonical.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
