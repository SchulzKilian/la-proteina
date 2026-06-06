#!/bin/bash
# Generate + fold one w of the L500 burial unlock test on one GPU. Resume-safe.
#   W=32|64  CUDA=<gpu>  -> b0 + b025 at L500, seeds 42-57, then fold codesign.
set -o pipefail
cd /home/ks2218/la-proteina
export PATH=$HOME/.conda/envs/laproteina_env/bin:$PATH
export MPNN_INPROCESS=1
PY=$HOME/.conda/envs/laproteina_env/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA:?}
W=${W:?}; SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"; ROOT=results/burial_camsol_probe
echo "[L500 gen] w=$W gpu=$CUDA_VISIBLE_DEVICES"
for tag in b0 b025; do
  echo "[$(date -u +%FT%TZ)] L500 w$W $tag"
  "$PY" -m steering.generate --proteina_config inference_ucond_notri_long \
      --steering_config steering/config/sweep_burial_camsol/camsol_w${W}_${tag}.yaml \
      --lengths 500 --seeds $SEEDS --nsteps 400 --skip_unguided --resume \
      --output_dir "$ROOT/L500_w${W}_${tag}" --device cuda:0
done
echo "[$(date -u +%FT%TZ)] folding L500 w$W"
$PY script_utils/run_burial_l500_eval.py fold "$ROOT/L500_w${W}_b0" "$ROOT/L500_w${W}_b025"
echo "[$(date -u +%FT%TZ)] === L500_w${W}_DONE ==="
