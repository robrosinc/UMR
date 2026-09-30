#!/usr/bin/env bash
# Open the configured OmniContact-to-Igris C result, including its object metadata.
# Optional arguments are forwarded to humanoid_retarget_pipeline_hsi_hoi.py.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="${ROOT}/../miniconda3/envs/UMR/bin/python"
PYTHON="${UMR_PYTHON:-${DEFAULT_PYTHON}}"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python)"
fi

exec "${PYTHON}" "${ROOT}/scripts/humanoid_retarget_pipeline_hsi_hoi.py" \
  --config "${ROOT}/robot_configs/humanoid_retarget_igris_c_omnicontact_example.json" \
  --defaults "${ROOT}/humanoid_retarget_defaults_hsi_hoi_standard.json" \
  --stage view \
  "$@"
