#!/usr/bin/env bash
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python

# Extension seeds for the K=1/K=5 w=32 cells (existing have 42-57; need 58-73 for n=32 at L=300)
EXTEND_SEEDS="58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73"
# Full 32-seed set for the fresh w=64 cells
ALL_SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73"

# --- 1. Extend K=1 w=32 baseline with seeds 58-73 ---
echo "[1/8 gen K1_w32 extend] $(date -Iseconds)"
$PY -m steering.generate \
  --proteina_config inference_ucond_notri_long \
  --steering_config steering/config/fixt1_smoke/tango_min_w32_fixt1_ensemble_denoised_n48.yaml \
  --lengths 300 --seeds $EXTEND_SEEDS \
  --output_dir results/fixt1_smoke/tango_min_w32_fixt1_ensemble_denoised_n48 \
  --device cuda:0 --skip_unguided

# --- 2. Extend K=5 w=32 with seeds 58-73 ---
echo "[2/8 gen K5_w32 extend] $(date -Iseconds)"
$PY -m steering.generate \
  --proteina_config inference_ucond_notri_long \
  --steering_config steering/config/ablation_2026_05_17/ug_K5.yaml \
  --lengths 300 --seeds $EXTEND_SEEDS \
  --output_dir results/ablation_2026_05_17/ug_K5 \
  --device cuda:0 --skip_unguided

# --- 3. Fresh K=1 w=64 at n=32 ---
echo "[3/8 gen K1_w64] $(date -Iseconds)"
mkdir -p results/ablation_2026_05_17/K1_w64
$PY -m steering.generate \
  --proteina_config inference_ucond_notri_long \
  --steering_config steering/config/ablation_2026_05_17/K1_w64.yaml \
  --lengths 300 --seeds $ALL_SEEDS \
  --output_dir results/ablation_2026_05_17/K1_w64 \
  --device cuda:0 --skip_unguided

# --- 4. Fresh K=5 w=64 at n=32 ---
echo "[4/8 gen K5_w64] $(date -Iseconds)"
mkdir -p results/ablation_2026_05_17/K5_w64
$PY -m steering.generate \
  --proteina_config inference_ucond_notri_long \
  --steering_config steering/config/ablation_2026_05_17/K5_w64.yaml \
  --lengths 300 --seeds $ALL_SEEDS \
  --output_dir results/ablation_2026_05_17/K5_w64 \
  --device cuda:0 --skip_unguided

# --- Codesign all 4 cells (resume-safe; only does new PDBs) ---
echo "[5/8 codesign K1_w32 extend] $(date -Iseconds)"
OUT_BASE=results/fixt1_smoke $PY scripts/run_codesignability_sweep.py \
  --lengths 300 --seeds $EXTEND_SEEDS \
  --cfgs tango_min_w32_fixt1_ensemble_denoised_n48

echo "[6/8 codesign K5_w32 extend] $(date -Iseconds)"
OUT_BASE=results/ablation_2026_05_17 $PY scripts/run_codesignability_sweep.py \
  --lengths 300 --seeds $EXTEND_SEEDS \
  --cfgs ug_K5

echo "[7/8 codesign K1_w64] $(date -Iseconds)"
OUT_BASE=results/ablation_2026_05_17 $PY scripts/run_codesignability_sweep.py \
  --lengths 300 --seeds $ALL_SEEDS \
  --cfgs K1_w64

echo "[8/8 codesign K5_w64] $(date -Iseconds)"
OUT_BASE=results/ablation_2026_05_17 $PY scripts/run_codesignability_sweep.py \
  --lengths 300 --seeds $ALL_SEEDS \
  --cfgs K5_w64

echo "[summary] $(date -Iseconds)"
$PY scripts/summary_kw_2x2.py

echo "ALL_DONE_KW $(date -Iseconds)"
