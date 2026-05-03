#!/bin/bash
# Steering sweep: camsol_intrinsic (maximize) and tango (minimize) at lengths 300/400/500.
#
# Always --skip_unguided: user already has a 1000-protein stratified unguided
# control at results/generated_stratified_300_800/. Compare distributionally.
#
# Uses the OFFICIAL La-Proteina LD+AE (inference_ucond_notri_long ->
# LD3_ucond_notri_800 + AE2_ucond_800). Required because the steering hook
# in product_space_flow_matcher.py only fires when 'local_latents' is in
# nn_out — CA-only baselines silently no-op the steering call.
#
# AE2 was the AE used to compute the predictor's training latents
# (data/pdb_train/processed_latents_300_800), so the latent space at sampling
# matches the predictor's input distribution.

set -uo pipefail

cd /home/ks2218/la-proteina

OUT=results/steering_camsol_tango_L500
LENGTHS="300 400 500"
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
PROTEINA_CFG=inference_ucond_notri_long
NSTEPS=100
DEVICE=cuda:0

mkdir -p "$OUT"
START=$(date +%s)
echo "[$(date)] Sweep started. Output: $OUT"
echo "[$(date)] LD/AE config: $PROTEINA_CFG  nsteps=$NSTEPS  device=$DEVICE"
echo "[$(date)] lengths=[$LENGTHS]  seeds=[$SEEDS]"

CFGS=(
    camsol_max_w1 camsol_max_w2 camsol_max_w4 camsol_max_w8 camsol_max_w16
    tango_min_w1 tango_min_w2 tango_min_w4 tango_min_w8 tango_min_w16
)

i=0
N=${#CFGS[@]}
for cfg in "${CFGS[@]}"; do
    i=$((i + 1))
    echo ""
    echo "=========================================================="
    echo "[$(date)] [${i}/${N}] ${cfg} (guided only, --skip_unguided)"
    echo "=========================================================="
    python -m steering.generate \
        --proteina_config "$PROTEINA_CFG" \
        --steering_config "steering/config/sweep_camsol_tango_L500/${cfg}.yaml" \
        --lengths $LENGTHS \
        --seeds $SEEDS \
        --nsteps "$NSTEPS" \
        --output_dir "$OUT/${cfg}" \
        --device "$DEVICE" \
        --skip_unguided
done

END=$(date +%s)
echo ""
echo "[$(date)] Sweep complete. Total: $(( (END-START)/60 )) min."
