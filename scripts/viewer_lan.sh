#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/robros/workspace/miniconda3/envs/UMR/bin/python"
LAN_HOST="$(ip -4 route get 1.1.1.1 | awk '{for (i = 1; i <= NF; i++) if ($i == "src") {print $(i + 1); exit}}')"
if [[ -z "$LAN_HOST" ]]; then
    echo "Could not determine the local network IP address." >&2
    exit 1
fi

echo "UMR LAN viewer: http://${LAN_HOST}:8081"
exec "$PYTHON" "$ROOT/scripts/visualize_robot_retarget_result.py" \
    --lan-viewer --viser-host "$LAN_HOST" --viser-port 8081 \
    --viser-label "UMR LAN Viewer" --play
