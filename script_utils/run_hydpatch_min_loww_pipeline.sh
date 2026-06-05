#!/usr/bin/env bash
# Low-w extension of the hydrophobic_patch_total_area MINIMIZE sweep.
#
# Mirrors the camsol/tango dose-response grid (run_noise_aware_ensemble_sweep.sh,
# WLEVELS=1 2 4 8 16) so hydpatch_min has the same low-w foot of the curve as
# camsol_max / tango_min. The existing run_hydpatch_min_pipeline.sh already
# covers the high-w head {32,48,64,128}; this fills in {1,2,4,8,16}. Same
# OUT_ROOT so steering_cost_audit.py covers all 9 w-levels in one tree.
#
# Canonical steering recipe (identical to run_hydpatch_min_pipeline.sh):
#   NA-v1 5-fold ensemble predictor, inference_ucond_notri_long,
#   linear_ramp schedule (t_start=0.3, t_end=0.8, t_stop=0.9), unit grad-norm,
#   nsteps=400, 16 seeds x L in {300,400,500}, w in {1,2,4,8,16} = 240 PDBs.
#
# Usage:
#   nohup bash script_utils/run_hydpatch_min_loww_pipeline.sh \
#       > nohup_hydpatch_min_loww.out 2>&1 &
set -uo pipefail
cd /home/ks2218/la-proteina

# Pin to free GPU 7 (session default GPU 6 is occupied by another user).
export CUDA_VISIBLE_DEVICES=7

PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
export IUPRED3_DIR=${IUPRED3_DIR:-/home/ks2218/iupred3}

SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400 500"
NSTEPS=400
DEVICE=cuda:0
OUT_ROOT=results/hydpatch_min_sweep
WLEVELS=(1 2 4 8 16)

mkdir -p "$OUT_ROOT"
echo "[$(date -u +%FT%TZ)] === hydpatch_min low-w (w 1/2/4/8/16) sweep starting on GPU 7 ==="
echo "5 w-levels x 48 PDBs each = 240 generations, single-objective steering."
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
echo "[$(date -u +%FT%TZ)] === all low-w cells generated; running audit over full tree (all 9 w-levels) ==="
"$PY" script_utils/steering_cost_audit.py \
    --tree "$OUT_ROOT" \
    --evals property,aa,codesign,diversity

echo
echo "[$(date -u +%FT%TZ)] === hydpatch_min low-w pipeline complete ==="
