#!/usr/bin/env bash
# Full pipeline for the w=48 knee-pinpointing experiment: generate camsol_max
# and tango_min at w=48 (16 seeds × 3 lengths each = 96 PDBs total), then run
# the audit chain (property, AA, codesign, diversity). Chained on a single GPU.
#
# Designed to be launched once under nohup and left to run unattended.
#
# Usage:
#   nohup bash script_utils/run_w48_full_pipeline.sh > nohup_w48_pipeline.out 2>&1 &
set -uo pipefail
cd /home/ks2218/la-proteina

PY=/home/ks2218/.conda/envs/laproteina_env/bin/python

echo "[$(date)] === phase 1: generate camsol_max_w48 ==="
bash script_utils/run_noise_aware_high_w_scout.sh camsol_max cuda:0 "48"

echo "[$(date)] === phase 2: generate tango_min_w48 ==="
bash script_utils/run_noise_aware_high_w_scout.sh tango_min  cuda:0 "48"

echo "[$(date)] === phase 3: full audit (property+aa+codesign+diversity) ==="
# audit script is resume-safe per-cell: existing CSVs in w=32/64/128 cells are
# skipped, only the new w=48 cells get property + codesign run. AA composition
# and diversity are rebuilt over the whole tree (fast).
"$PY" script_utils/steering_cost_audit.py \
    --tree results/noise_aware_high_w_scout \
    --evals property,aa,codesign,diversity

echo "[$(date)] === w=48 pipeline complete ==="
