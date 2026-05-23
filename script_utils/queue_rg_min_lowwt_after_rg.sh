#!/usr/bin/env bash
# Queue the low-w Rg-min coord-channel sweep behind the running Rg-min
# {32,48,64} main sweep. Tests the codesign-preserving end of the
# dose-response curve.
set -uo pipefail
cd /home/ks2218/la-proteina

WAIT_PID=${1:?"usage: $0 <wait-PID> [cuda:N=cuda:0]"}
DEVICE=${2:-cuda:0}

echo "[$(date)] waiting for Rg-min main sweep PID $WAIT_PID ..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[$(date)] PID $WAIT_PID exited. launching low-w Rg-min coord-channel sweep (w∈{4,8,16})."

bash script_utils/run_coords_na_rg_min_coordonly.sh "$DEVICE" "4 8 16"
echo "[$(date)] low-w Rg-min coord-channel sweep done."
