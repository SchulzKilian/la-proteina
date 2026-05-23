#!/usr/bin/env bash
# 4-objective developability cocktail steering scout:
#   camsol_intrinsic max + tango min + sap min + scm_positive min, all w=1.0,
# w ∈ {32, 48, 64, 128} × 16 seeds × 3 lengths each (= 192 PDBs total).
#
# Tests E068's open question "what's the maximum number of objectives before
# gradients destructively interfere?" by extending the 2-obj combo to 4
# correlated-but-mechanistically-distinct anti-aggregation objectives.
# Two new axes added vs combo_camsol_tango: SAP (3D-surface aggregation) and
# SCM_positive (3D-surface charge clustering, the antibody high-concentration
# viscosity / polyspecificity axis).
#
# Audit chain: property + AA + codesign + diversity (resume-safe per cell).
#
# Usage:
#   nohup bash script_utils/run_combo_devel4_pipeline.sh \
#       > nohup_combo_devel4.out 2>&1 &
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
OUT_ROOT=results/combo_devel4_scout
WLEVELS=(32 48 64 128)

mkdir -p "$OUT_ROOT"
echo "[$(date -u +%FT%TZ)] === combo_devel4 (camsol_max+tango_min+sap_min+scm_positive_min) scout starting ==="
echo "4 w-levels × 48 PDBs each = 192 generations, 4-objective steering."
echo

# Step 1: generate
for w in "${WLEVELS[@]}"; do
    cfg="combo_devel4_w${w}"
    out="$OUT_ROOT/$cfg"
    # Resume-safe: skip generation if guided/ already has 48 PDBs
    n_pdb=$(ls "$out/guided"/*.pdb 2>/dev/null | wc -l)
    if [ "$n_pdb" -ge 48 ]; then
        echo "[$(date -u +%FT%TZ)] [$DEVICE] $cfg already has $n_pdb PDBs; skipping generation"
        continue
    fi
    echo "[$(date -u +%FT%TZ)] [$DEVICE] starting $cfg generation"
    "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long \
        --steering_config "steering/config/sweep_combo_devel4/${cfg}.yaml" \
        --lengths $LENGTHS \
        --seeds $SEEDS \
        --nsteps $NSTEPS \
        --skip_unguided \
        --output_dir "$out" \
        --device $DEVICE
    echo "[$(date -u +%FT%TZ)] [$DEVICE] finished $cfg generation"
done

echo
echo "[$(date -u +%FT%TZ)] === all 4 cells generated; starting audit (property + aa + codesign + diversity) ==="
"$PY" script_utils/steering_cost_audit.py \
    --tree "$OUT_ROOT" \
    --evals property,aa,codesign,diversity

echo
echo "[$(date -u +%FT%TZ)] === combo_devel4 pipeline complete ==="
