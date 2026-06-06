#!/bin/bash
# Confirmation run for the burial throttle (E114-add-9): (1) MORE SEEDS at w64
# (58-73 -> n=64/cell) to firm up the p=0.021 result; (2) HIGHER w (w128, baseline
# only 2% codesign) to test the strongest demonstration. b0 (no throttle) vs b025
# (burial beta=0.25). Resume-safe; reuses existing w64 seeds 42-57.
set -o pipefail
cd /home/ks2218/la-proteina
export PATH=$HOME/.conda/envs/laproteina_env/bin:$PATH
export MPNN_INPROCESS=1
PY=$HOME/.conda/envs/laproteina_env/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA:-3}
NEW_SEEDS="58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73"
W128_SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400"
ROOT=results/burial_camsol_probe
echo "[burial MORE] gpu=$CUDA_VISIBLE_DEVICES nsteps=400"

# (1) w64 extra seeds for b0 + b025 (appended into existing dirs via --resume)
for tag in b0 b025; do
  echo "[$(date -u +%FT%TZ)] w64 +seeds $tag"
  "$PY" -m steering.generate --proteina_config inference_ucond_notri_long \
      --steering_config steering/config/sweep_burial_camsol/camsol_w64_${tag}.yaml \
      --lengths $LENGTHS --seeds $NEW_SEEDS --nsteps 400 --skip_unguided --resume \
      --output_dir "$ROOT/$tag" --device cuda:0
done

# (2) w128 (higher w) b0 + b025
for tag in b0 b025; do
  echo "[$(date -u +%FT%TZ)] w128 $tag"
  "$PY" -m steering.generate --proteina_config inference_ucond_notri_long \
      --steering_config steering/config/sweep_burial_camsol/camsol_w128_${tag}.yaml \
      --lengths $LENGTHS --seeds $W128_SEEDS --nsteps 400 --skip_unguided --resume \
      --output_dir "$ROOT/${tag}_w128" --device cuda:0
done

echo "[$(date -u +%FT%TZ)] === EVAL2 (surfKD frontier + matched-band Fisher) ==="
"$PY" script_utils/run_burial_camsol_eval2.py
echo "[$(date -u +%FT%TZ)] === BURIAL_MORE_DONE ==="
