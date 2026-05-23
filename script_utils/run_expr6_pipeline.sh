#!/usr/bin/env bash
# 6-objective expression-feasibility cocktail steering Pareto run:
#   camsol_intrinsic max + tango min + sap min +
#   hydrophobic_patch_total_area min + scm_positive min +
#   iupred3 min (weight 0.5; others 1.0).
# Pareto sweep: w_max ∈ {16, 32, 48, 64} × 16 seeds × 3 lengths = 192 PDBs.
#
# Extends E072's 4-obj combo_devel4 with two additional anti-aggregation
# axes:
#   - hydrophobic_patch_total_area_min: orthogonal angle on surface
#     hydrophobicity (SAP is SASA-weighted; patch_area is raw surface area).
#   - iupred3_min (weight 0.5): counters the +0.65σ disorder drift observed
#     in combo_devel4_w32 at L=300. Half-weight to avoid fighting tango's
#     helix-promoter chemistry.
#
# Audit chain: property + AA + codesign + diversity.
#
# Usage:
#   nohup bash script_utils/run_expr6_pipeline.sh > nohup_expr6.out 2>&1 &
set -uo pipefail
cd /home/ks2218/la-proteina

PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
export IUPRED3_DIR=${IUPRED3_DIR:-/home/ks2218/iupred3}

SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400 500"
NSTEPS=400
DEVICE=${1:-cuda:1}
OUT_ROOT=results/expr6_pareto
WLEVELS=(16 32 48 64)

mkdir -p "$OUT_ROOT"
echo "[$(date -u +%FT%TZ)] === expr6 6-obj expression-feasibility Pareto starting on $DEVICE ==="
echo "objectives: camsol_max + tango_min + sap_min + hpa_min + scm_pos_min + iupred_min(w=0.5)"
echo "4 w-levels × 48 PDBs each = 192 generations."
echo

# Step 1: generate
for w in "${WLEVELS[@]}"; do
    cfg="expr6_w${w}"
    out="$OUT_ROOT/$cfg"
    n_pdb=$(ls "$out/guided"/*.pdb 2>/dev/null | wc -l)
    if [ "$n_pdb" -ge 48 ]; then
        echo "[$(date -u +%FT%TZ)] [$DEVICE] $cfg already has $n_pdb PDBs; skipping generation"
        continue
    fi
    echo "[$(date -u +%FT%TZ)] [$DEVICE] starting $cfg generation"
    "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long \
        --steering_config "steering/config/sweep_expr6/${cfg}.yaml" \
        --lengths $LENGTHS \
        --seeds $SEEDS \
        --nsteps $NSTEPS \
        --skip_unguided \
        --output_dir "$out" \
        --device $DEVICE
    echo "[$(date -u +%FT%TZ)] [$DEVICE] finished $cfg generation"
done

echo
echo "[$(date -u +%FT%TZ)] === all 4 cells generated; starting audit ==="
"$PY" script_utils/steering_cost_audit.py \
    --tree "$OUT_ROOT" \
    --evals property,aa,codesign,diversity

echo
echo "[$(date -u +%FT%TZ)] === expr6 pipeline complete ==="
