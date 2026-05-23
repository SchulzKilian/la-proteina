#!/usr/bin/env bash
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"

echo "[folds gen] $(date -Iseconds)"
GROUP=folds /home/ks2218/la-proteina/scripts/run_ablation_2026_05_17.sh

echo "[folds codesign] $(date -Iseconds)"
OUT_BASE=results/ablation_2026_05_17 $PY scripts/run_codesignability_sweep.py \
  --lengths 300 --seeds $SEEDS \
  --cfgs fold3

echo "[folds summary] $(date -Iseconds)"
$PY scripts/eval_ablation_2026_05_17.py | grep -E "fold3|baseline|n="

echo "ALL_DONE_FOLDS $(date -Iseconds)"
