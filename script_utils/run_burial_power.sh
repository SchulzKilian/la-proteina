#!/bin/bash
# Power run: +seeds to b0/b025 at w64, L300+L400, then fold. env: CUDA, COND(b0|b025), SEEDS
set -o pipefail; cd /home/ks2218/la-proteina
export PATH=$HOME/.conda/envs/laproteina_env/bin:$PATH; export MPNN_INPROCESS=1
PY=$HOME/.conda/envs/laproteina_env/bin/python; export CUDA_VISIBLE_DEVICES=${CUDA:?}
$PY -m steering.generate --proteina_config inference_ucond_notri_long \
  --steering_config steering/config/sweep_burial_camsol/camsol_w64_${COND:?}.yaml \
  --lengths 300 400 --seeds ${SEEDS:?} --nsteps 400 --skip_unguided --resume \
  --output_dir results/burial_camsol_probe/${COND} --device cuda:0
$PY script_utils/run_burial_camsol_eval2.py fold results/burial_camsol_probe/${COND}
echo "=== POWER_${COND}_DONE ==="
