#!/usr/bin/env bash
# Full noise-aware-ensemble sweep — apples-to-apples vs E028's
# sweep_camsol_tango_ensemble_smoothed grid: same 16 seeds × 3 lengths × 5 w
# values × 2 directions. Difference: 5 v1 noise-aware ckpts (E029) instead of
# clean ckpts; smoothing OFF (clean "noise-aware + ensemble alone" claim).
set -uo pipefail
cd /home/ks2218/la-proteina

PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400 500"
NSTEPS=400
DEVICE=cuda:0
OUT_ROOT=results/noise_aware_ensemble_sweep
DIRECTIONS=(camsol_max tango_min)
WLEVELS=(1 2 4 8 16)

mkdir -p $OUT_ROOT
echo "[$(date)] sweep started. Output: $OUT_ROOT"
echo "10 cells (2 directions × 5 w-levels) × 48 PDBs each = 480 generations."
echo

idx=1
total=$((${#DIRECTIONS[@]} * ${#WLEVELS[@]}))

for dir in "${DIRECTIONS[@]}"; do
  for w in "${WLEVELS[@]}"; do
    cfg="${dir}_w${w}"
    out="$OUT_ROOT/$cfg"
    echo "[$(date)] [$idx/$total] $cfg"
    $PY -m steering.generate \
      --proteina_config inference_ucond_notri_long \
      --steering_config "steering/config/sweep_noise_aware_ensemble/${cfg}.yaml" \
      --lengths $LENGTHS \
      --seeds $SEEDS \
      --nsteps $NSTEPS \
      --skip_unguided \
      --output_dir "$out" \
      --device $DEVICE
    idx=$((idx + 1))
  done
done

echo
echo "[$(date)] sweep complete."
