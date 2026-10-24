#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${GRAIL_INPUT_DIR:-$ROOT/sample_data/grail/stair_p2}"
OUTPUT_DIR="${GRAIL_OUTPUT_DIR:-$ROOT/output/igris_c_grail}"
PYTHON_BIN="${UMR_PYTHON:-/home/robros/workspace/miniconda3/envs/UMR/bin/python}"

if (( $# > 0 )) && [[ "$1" != -* ]]; then
  INPUT_DIR="$1"
  shift
fi
if (( $# > 0 )) && [[ "$1" != -* ]]; then
  OUTPUT_DIR="$1"
  shift
fi

exec "$PYTHON_BIN" "$ROOT/scripts/humanoid_retarget_pipeline_hsi_hoi_batch.py" \
  --config "$ROOT/robot_configs/humanoid_retarget_igris_c_grail.json" \
  --defaults "$ROOT/humanoid_retarget_defaults_hsi_hoi_grail.json" \
  --data "$INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --workers 24 \
  "$@"
