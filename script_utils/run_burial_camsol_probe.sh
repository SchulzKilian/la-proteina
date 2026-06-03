#!/bin/bash
# Burial-throttle camsol probe: does protecting the buried core during camsol_max
# steering keep designability up while still delivering solubility? A/B at w64:
#   b0  = no throttle (matched-seed baseline)
#   b010 = burial throttle beta=0.10
#   b025 = burial throttle beta=0.25
# Then eval codesign + tool-free delivery proxy (KD hydrophobicity) vs the existing
# no-throttle camsol frontier (noise_aware_high_w w32/48/64/128). Resume-safe.
set -o pipefail
cd /home/ks2218/la-proteina
export PATH=$HOME/.conda/envs/laproteina_env/bin:$PATH
export MPNN_INPROCESS=1
PY=$HOME/.conda/envs/laproteina_env/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA:-3}
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57}"
LENGTHS="${LENGTHS:-300 400}"
ROOT=results/burial_camsol_probe
echo "[burial probe] gpu=$CUDA_VISIBLE_DEVICES seeds='$SEEDS' L='$LENGTHS' nsteps=400"

for tag in b0 b010 b025; do
  cfg=steering/config/sweep_burial_camsol/camsol_w64_${tag}.yaml
  echo "[$(date -u +%FT%TZ)] generating $tag ($cfg)"
  "$PY" -m steering.generate \
      --proteina_config inference_ucond_notri_long --steering_config "$cfg" \
      --lengths $LENGTHS --seeds $SEEDS --nsteps 400 --skip_unguided --resume \
      --output_dir "$ROOT/$tag" --device cuda:0
done

echo "[$(date -u +%FT%TZ)] === EVAL (codesign + KD frontier) ==="
"$PY" script_utils/run_burial_camsol_eval.py
echo "[$(date -u +%FT%TZ)] === BURIAL_PROBE_DONE ==="
