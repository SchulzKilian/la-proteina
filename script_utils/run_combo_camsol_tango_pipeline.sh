#!/usr/bin/env bash
# Multi-objective steering scout: camsol_max + tango_min simultaneously,
# w ∈ {16, 32, 64, 128} × 16 seeds × 3 lengths each (= 192 PDBs).
# Audit chain: property + AA + codesign + diversity.
#
# Usage:
#   nohup bash script_utils/run_combo_camsol_tango_pipeline.sh \
#       > nohup_combo_camsol_tango.out 2>&1 &
set -uo pipefail
cd /home/ks2218/la-proteina

PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400 500"
NSTEPS=400
DEVICE=cuda:0
OUT_ROOT=results/combo_camsol_tango_scout
WLEVELS=(16 32 64 128)

mkdir -p "$OUT_ROOT"
echo "[$(date)] === combo camsol_max+tango_min scout starting ==="
echo "4 w-levels × 48 PDBs each = 192 generations, multi-objective steering."
echo

for w in "${WLEVELS[@]}"; do
    cfg="combo_camsol_tango_w${w}"
    out="$OUT_ROOT/$cfg"
    echo "[$(date)] [$DEVICE] starting $cfg"
    "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long \
        --steering_config "steering/config/sweep_combo_camsol_tango/${cfg}.yaml" \
        --lengths $LENGTHS \
        --seeds $SEEDS \
        --nsteps $NSTEPS \
        --skip_unguided \
        --output_dir "$out" \
        --device $DEVICE
    echo "[$(date)] [$DEVICE] finished $cfg"
done

echo
echo "[$(date)] === all 4 cells generated; starting audit ==="
"$PY" script_utils/steering_cost_audit.py \
    --tree "$OUT_ROOT" \
    --evals property,aa,codesign,diversity

echo
echo "[$(date)] === combo pipeline complete ==="
