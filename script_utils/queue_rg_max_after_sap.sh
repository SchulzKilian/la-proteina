#!/usr/bin/env bash
# Queue Rg-max coord-channel sweep behind the running SAP-min sweep.
set -uo pipefail
cd /home/ks2218/la-proteina

WAIT_PID=${1:?"usage: $0 <sap-sweep-PID> [cuda:N=cuda:0]"}
DEVICE=${2:-cuda:0}

echo "[$(date)] waiting for SAP coord-channel sweep PID $WAIT_PID ..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[$(date)] PID $WAIT_PID exited. launching Rg-max coord-channel sweep."

bash script_utils/run_coords_na_rg_coordonly.sh "$DEVICE" "32 48 64"
echo "[$(date)] Rg-max coord-channel sweep done."
