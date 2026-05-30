#!/usr/bin/env bash
# Single-objective steering sweep: hydrophobic_patch_total_area MINIMIZE.
#
# Therapeutic rationale: smaller exposed hydrophobic surface patches -> lower
# aggregation propensity / better solubility / better developability. This is
# the first standalone sweep of the hydrophobic_patch_total_area axis (E072
# only exercised SAP, and only inside a 4-objective cocktail).
#
# Canonical steering recipe (matches E066/E068/E072):
#   NA-v1 5-fold ensemble predictor, inference_ucond_notri_long,
#   linear_ramp schedule (t_start=0.3, t_end=0.8, t_stop=0.9), unit grad-norm,
#   nsteps=400, 16 seeds x L in {300,400,500}, w in {32,48,64,128} = 192 PDBs.
#
# Audit chain: property + AA + codesign + diversity (resume-safe per cell).
#
# Usage:
#   nohup bash script_utils/run_hydpatch_min_pipeline.sh \
#       > nohup_hydpatch_min.out 2>&1 &
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
OUT_ROOT=results/hydpatch_min_sweep
WLEVELS=(32 48 64 128)

mkdir -p "$OUT_ROOT"
echo "[$(date -u +%FT%TZ)] === hydpatch_min (hydrophobic_patch_total_area minimize) sweep starting ==="
echo "4 w-levels x 48 PDBs each = 192 generations, single-objective steering."
echo

# Step 1: generate
for w in "${WLEVELS[@]}"; do
    cfg="hydpatch_min_w${w}"
    out="$OUT_ROOT/$cfg"
    n_pdb=$(ls "$out/guided"/*.pdb 2>/dev/null | wc -l)
    if [ "$n_pdb" -ge 48 ]; then
        echo "[$(date -u +%FT%TZ)] [$DEVICE] $cfg already has $n_pdb PDBs; skipping generation"
        continue
    fi
    echo "[$(date -u +%FT%TZ)] [$DEVICE] starting $cfg generation"
    "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long \
        --steering_config "steering/config/sweep_hydpatch_min/${cfg}.yaml" \
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
echo "[$(date -u +%FT%TZ)] === hydpatch_min pipeline complete ==="
