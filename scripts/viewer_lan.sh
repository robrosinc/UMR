#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIEWER_LABEL="${UMR_VIEWER_LABEL:-UMR LAN Viewer}"
MAX_CLIENTS="${UMR_VIEWER_MAX_CLIENTS:-6}"
LAN_HOST="${UMR_VIEWER_HOST:-$(ip -4 route get 1.1.1.1 | awk '{for (i = 1; i <= NF; i++) if ($i == "src") {print $(i + 1); exit}}')}"
if [[ -z "$LAN_HOST" ]]; then
    echo "Could not determine the local network IP address." >&2
    exit 1
fi

echo "UMR LAN viewer: http://${LAN_HOST}:8081"
exec python "$ROOT/scripts/visualize_robot_retarget_result.py" \
    --lan-viewer --viser-host "$LAN_HOST" --viser-port 8081 \
    --viewer-max-clients "$MAX_CLIENTS" \
    --viser-label "$VIEWER_LABEL" --play
