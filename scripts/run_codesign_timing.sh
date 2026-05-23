#!/usr/bin/env bash
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"

echo "[codesign timing cells] $(date -Iseconds)"
OUT_BASE=results/ablation_2026_05_17 $PY scripts/run_codesignability_sweep.py \
  --lengths 300 --seeds $SEEDS \
  --cfgs constant_all late_only early_only wide_ramp

echo "[codesign fixt1 baseline] $(date -Iseconds)"
OUT_BASE=results/fixt1_smoke $PY scripts/run_codesignability_sweep.py \
  --lengths 300 --seeds $SEEDS \
  --cfgs tango_min_w32_fixt1_ensemble_denoised_n48

echo "[summary] $(date -Iseconds)"
$PY scripts/summary_codesign_timing.py

echo "ALL_DONE $(date -Iseconds)"
