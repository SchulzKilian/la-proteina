#!/usr/bin/env bash
# Wait for the net_charge_min audit to finish, then launch the overnight
# schedule + length-extension pipeline.
set -uo pipefail

DONE_MARKER=/home/ks2218/la-proteina/results/net_charge_min_scout/diversity_pairwise_tm.csv

echo "[$(date)] waiting for net_charge_min audit to finish..."
while true; do
    if [[ -f "$DONE_MARKER" ]]; then
        if ! pgrep -f "steering_cost_audit\|codesign_sweep\|diversity_pairwise\|run_net_charge_pareto" > /dev/null; then
            echo "[$(date)] net_charge_min done; launching overnight pipeline"
            break
        fi
    fi
    sleep 60
done
sleep 30  # buffer for final flushes
exec bash /home/ks2218/la-proteina/script_utils/run_overnight_schedule_and_length.sh
