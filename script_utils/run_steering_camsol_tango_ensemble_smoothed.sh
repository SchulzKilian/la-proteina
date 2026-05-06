#!/bin/bash
# E025 follow-up: 5-fold ensemble + Gaussian-smoothed gradient. Same 10 configs,
# same 16 seeds × 3 lengths as the nsteps=400 regen. Single GPU on cuda:0.
# Estimated wall: ~4.3 h (32s/protein × 480, vs ~9s/protein for single-fold).

set -o pipefail
cd /home/ks2218/la-proteina
source /opt/conda/etc/profile.d/conda.sh
conda activate laproteina_env
set -u

OUT=results/steering_camsol_tango_L500_ensemble_smoothed
LENGTHS="300 400 500"
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
PROTEINA_CFG=inference_ucond_notri_long
NSTEPS=400
DEVICE=cuda:0

mkdir -p "$OUT"
START=$(date +%s)
echo "[$(date)] ensemble+smoothing sweep started. Output: $OUT"
echo "[$(date)] LD/AE: $PROTEINA_CFG  nsteps=$NSTEPS  device=$DEVICE  5-fold ensemble + sigma=0.1, K=4"

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
    echo "[$(date)] [${i}/${N}] ${cfg} (5-fold ensemble + smoothed, --skip_unguided)"
    echo "=========================================================="
    python -m steering.generate \
        --proteina_config "$PROTEINA_CFG" \
        --steering_config "steering/config/sweep_camsol_tango_ensemble_smoothed/${cfg}.yaml" \
        --lengths $LENGTHS \
        --seeds $SEEDS \
        --nsteps "$NSTEPS" \
        --output_dir "$OUT/${cfg}" \
        --device "$DEVICE" \
        --skip_unguided
done

END=$(date +%s)
echo ""
echo "[$(date)] ensemble+smoothing sweep complete. Total wall: $(( (END-START)/60 )) min."
