#!/usr/bin/env bash
# Property + AA audit on the fixt1 Pareto-frontier rerun cells.
# Runs sequentially through E066/E067/E068; resume-safe per cell.
set -uo pipefail

cd /home/ks2218/la-proteina
# Use the laproteina env's python directly — same one the codesign chain uses.
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python

export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
export IUPRED3_DIR=${IUPRED3_DIR:-/home/ks2218/iupred3}

ROOT=/home/ks2218/la-proteina/results/fixt1_full_replication_2026_05_18
LOG=$ROOT/property_audit.log

echo "=== START $(date -u +%FT%TZ) ===" | tee -a "$LOG"

for tree in "$ROOT/E066_high_w" "$ROOT/E067_iupred_max" "$ROOT/E068_combo_camsol_tango"; do
    echo "=== tree: $tree ===" | tee -a "$LOG"
    "$PY" script_utils/steering_cost_audit.py --tree "$tree" --evals property,aa \
        2>&1 | tee -a "$LOG"
    echo "=== done tree: $tree at $(date -u +%FT%TZ) ===" | tee -a "$LOG"
done

echo "=== END $(date -u +%FT%TZ) ===" | tee -a "$LOG"
