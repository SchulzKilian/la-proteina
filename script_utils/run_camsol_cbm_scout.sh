#!/usr/bin/env bash
# E110: camsol_intrinsic MAXIMIZE Pareto driven by the CONCEPT-BOTTLENECK predictor (best CBM fold_2),
# replicating E066 (NA-v1 ensemble) so the codesign-vs-SWI frontier is directly comparable.
# Real camsol is NaN at eval -> the measured solubility axis is SWI (developability), as in E066.
# w in {16,32,64,128} x 16 seeds x L{300,400,500} = 192 PDBs, nsteps=400, inference_ucond_notri_long.
#
# Usage: nohup bash script_utils/run_camsol_cbm_scout.sh > nohup_camsol_cbm.out 2>&1 &
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
export IUPRED3_DIR=${IUPRED3_DIR:-/home/ks2218/iupred3}

SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400 500"
NSTEPS=400
DEVICE=cuda:0
OUT_ROOT=results/camsol_cbm_scout
WLEVELS=(16 32 64 128)
mkdir -p "$OUT_ROOT"

echo "[$(date -u +%FT%TZ)] === CBM-driven camsol_max Pareto (w 16/32/64/128) ==="
for w in "${WLEVELS[@]}"; do
    cfg="camsol_max_w${w}"
    out="$OUT_ROOT/$cfg"
    n_pdb=$(ls "$out/guided"/*.pdb 2>/dev/null | wc -l)
    if [ "$n_pdb" -ge 48 ]; then
        echo "[$(date -u +%FT%TZ)] $cfg already has $n_pdb PDBs; skipping gen"; continue
    fi
    echo "[$(date -u +%FT%TZ)] [$DEVICE] generating $cfg"
    "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long \
        --steering_config "steering/config/sweep_camsol_cbm/${cfg}.yaml" \
        --lengths $LENGTHS --seeds $SEEDS --nsteps $NSTEPS --skip_unguided \
        --output_dir "$out" --device $DEVICE
done

echo "[$(date -u +%FT%TZ)] === audit (SWI property + codesign + diversity) ==="
"$PY" script_utils/steering_cost_audit.py --tree "$OUT_ROOT" --evals property,codesign,diversity
echo "[$(date -u +%FT%TZ)] === camsol CBM scout complete ==="
