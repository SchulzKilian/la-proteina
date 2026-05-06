#!/bin/bash
# Steering sweep regen at nsteps=400 (was nsteps=100 — too coarse, structures
# off-manifold per scRMSD sanity test 2026-05-03 where unguided nsteps=100 gave
# scRMSD ≈ 24 A vs nsteps=400 ≈ 0.8 A on the same model + seed).
#
# Same 10 configs / 16 seeds / 3 lengths / LD3+AE2 as the original sweep —
# only nsteps changes, plus output dir suffix _nsteps400 to keep results
# separate. SINGLE GPU on cuda:0 — sequential pass over the 10 configs.
# Estimated wall: ~5 h (4× the nsteps=100 sweep's 68 min).

set -o pipefail
cd /home/ks2218/la-proteina

# Activate conda env (required when launched via nohup — login shell init isn't sourced).
# Don't use `set -u` before activate.d scripts run: MKL_INTERFACE_LAYER is unbound.
source /opt/conda/etc/profile.d/conda.sh
conda activate laproteina_env
set -u

OUT=results/steering_camsol_tango_L500_nsteps400
LENGTHS="300 400 500"
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
PROTEINA_CFG=inference_ucond_notri_long
NSTEPS=400
DEVICE=cuda:0

mkdir -p "$OUT"
START=$(date +%s)
echo "[$(date)] nsteps=400 regen started. Output: $OUT"
echo "[$(date)] LD/AE: $PROTEINA_CFG  nsteps=$NSTEPS  device=$DEVICE"
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
echo "[$(date)] Sweep regen complete. Total wall: $(( (END-START)/60 )) min."
