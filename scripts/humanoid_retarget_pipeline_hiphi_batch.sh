#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${REPO_ROOT}/sample_data/hiphi"
OUTPUT_DIR="${REPO_ROOT}/output/igris_c_hiphi"
RETARGET_WORKERS=1
OBJECT_WORKERS=1
CONTACT_ONLY="${CONTACT_ONLY:-false}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
for arg in "$@"; do
    [[ "$arg" == "--contact-only" ]] && CONTACT_ONLY=true
done
if [[ "${CONTACT_ONLY}" == true ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}_contact_only"
    set -- --contact-only "$@"
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/humanoid_retarget_pipeline_hiphi_batch.py" \
    --data-root "${INPUT_DIR}" --output-root "${OUTPUT_DIR}" \
    --retarget-gpus 0 --stride 3 \
    --workers "${RETARGET_WORKERS}" --object-workers "${OBJECT_WORKERS}" "$@"
