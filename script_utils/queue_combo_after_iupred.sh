#!/usr/bin/env bash
# Wait for the iupred audit to finish, then launch the combo
# (camsol_max + tango_min) pipeline. Idempotent: polls every 60s.
set -uo pipefail

IUPRED_DIV=/home/ks2218/la-proteina/results/iupred_max_scout/diversity_pairwise_tm.csv

echo "[$(date)] waiting for iupred audit to finish..."
while true; do
    if [[ -f "$IUPRED_DIV" ]]; then
        if ! pgrep -f "steering_cost_audit\|codesign_sweep\|diversity_pairwise" > /dev/null; then
            echo "[$(date)] iupred audit done; launching combo pipeline"
            break
        fi
    fi
    sleep 60
done

# Small buffer to let any final flushes finish
sleep 30

exec bash /home/ks2218/la-proteina/script_utils/run_combo_camsol_tango_pipeline.sh
