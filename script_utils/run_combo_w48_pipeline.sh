#!/usr/bin/env bash
# combo_camsol_tango at w=48 only: generation + audit chain
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
echo "[$(date)] === combo_camsol_tango_w48 starting ==="
"$PY" -m steering.generate \
    --proteina_config inference_ucond_notri_long \
    --steering_config steering/config/sweep_combo_camsol_tango/combo_camsol_tango_w48.yaml \
    --lengths 300 400 500 --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 \
    --nsteps 400 --skip_unguided \
    --output_dir results/combo_camsol_tango_scout/combo_camsol_tango_w48 \
    --device cuda:0
echo "[$(date)] === generation done; running audit ==="
"$PY" script_utils/steering_cost_audit.py \
    --tree results/combo_camsol_tango_scout \
    --evals property,aa,codesign,diversity
echo "[$(date)] === w=48 combo pipeline complete ==="
