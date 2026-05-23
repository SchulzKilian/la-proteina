#!/usr/bin/env bash
# Full Pareto ablation for net_charge steering: net_charge_max AND net_charge_min,
# w ∈ {16, 32, 64, 128} × 16 seeds × 3 lengths each = 384 PDBs total.
# Generation + audit chain (property + AA + codesign + diversity) per direction.
set -uo pipefail
cd /home/ks2218/la-proteina

PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400 500"
NSTEPS=400
DEVICE=cuda:0
WLEVELS=(16 32 64 128)

for DIRECTION in net_charge_max net_charge_min; do
    OUT_ROOT="results/${DIRECTION}_scout"
    mkdir -p "$OUT_ROOT"
    echo "[$(date)] === starting ${DIRECTION} sweep ==="
    for w in "${WLEVELS[@]}"; do
        cfg="${DIRECTION}_w${w}"
        out="$OUT_ROOT/$cfg"
        echo "[$(date)] [$DEVICE] starting $cfg"
        "$PY" -m steering.generate \
            --proteina_config inference_ucond_notri_long \
            --steering_config "steering/config/sweep_${DIRECTION}/${cfg}.yaml" \
            --lengths $LENGTHS \
            --seeds $SEEDS \
            --nsteps $NSTEPS \
            --skip_unguided \
            --output_dir "$out" \
            --device $DEVICE
        echo "[$(date)] [$DEVICE] finished $cfg"
    done
    echo "[$(date)] === ${DIRECTION} generation complete; running audit ==="
    "$PY" script_utils/steering_cost_audit.py \
        --tree "$OUT_ROOT" \
        --evals property,aa,codesign,diversity
    echo "[$(date)] === ${DIRECTION} pipeline complete ==="
done

echo
echo "[$(date)] === BOTH net_charge sweeps complete ==="
