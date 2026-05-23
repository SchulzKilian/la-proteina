#!/usr/bin/env bash
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"

echo "[shape gen] $(date -Iseconds)"
GROUP=shape /home/ks2218/la-proteina/scripts/run_ablation_2026_05_17.sh

echo "[shape codesign] $(date -Iseconds)"
OUT_BASE=results/ablation_2026_05_17 $PY scripts/run_codesignability_sweep.py \
  --lengths 300 --seeds $SEEDS \
  --cfgs cosine constant_in_window

echo "[shape summary] $(date -Iseconds)"
$PY scripts/summary_shape.py

echo "ALL_DONE_SHAPE $(date -Iseconds)"
