#!/usr/bin/env bash
# Queue wrapper: wait for the n=64 L400 job (PID passed as $1) to finish freeing
# the GPU, then run the iupred_target sweep. Single-GPU discipline -> no overlap.
set -uo pipefail
cd /home/ks2218/la-proteina

WAIT_PID="${1:?usage: queue_iupred_after_n64.sh <pid_to_wait_for>}"

echo "[$(date -u +%FT%TZ)] queued: waiting for PID $WAIT_PID (n=64 L400 job) to finish before starting iupred_target"
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(date -u +%FT%TZ)] PID $WAIT_PID done; GPU free -> launching iupred_target pipeline"

exec bash script_utils/run_iupred_target_pipeline.sh
