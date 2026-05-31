#!/usr/bin/env bash
# Single-objective TARGET steering sweep: iupred3_fraction_disordered -> target_value = 0.123 (AFDB natural).
#
# La-Proteina unsteered is OVER-disordered (~0.158 fraction_disordered) vs AFDB
# natural 0.123 (E067). This steers DOWN to the natural setpoint -- the disorder
# analog of net_charge target=-5 (E105). E067 only MAXIMIZED disorder (-> IDP
# regime); this minimize-toward-natural direction is new. IUPred is the cleanest
# steerable axis (E067: predictor UNDER-promises, least gradient-hackable), and
# the move is small, so a cheap/"free" knee at low w is the hypothesis.
#
# Recipe IDENTICAL to E105 (net_charge_target) except the objective, so directly
# comparable: NA-v1 5-fold ensemble, inference_ucond_notri_long, linear_ramp
# (t_start=0.3, t_end=0.8, t_stop=0.9), unit grad-norm, denoised input, nsteps=400,
# 16 seeds x L in {300,400,500}, w in {8,16,24,32} = 192 PDBs.
#
# Designability read-out: pair against the seed-matched unguided baseline
# results/noise_aware_ensemble_sweep/codesign_unsteered_matched_seed.csv (E105).
#
# Usage:
#   nohup bash script_utils/run_iupred_target_pipeline.sh \
#       > nohup_iupred_target.out 2>&1 &
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
OUT_ROOT=results/iupred_target_sweep
WLEVELS=(8 16 24 32)

mkdir -p "$OUT_ROOT"
echo "[$(date -u +%FT%TZ)] === iupred3_fraction_disordered TARGET=0.123 sweep starting ==="
echo "4 w-levels x 48 PDBs each = 192 generations, single-objective target steering."
echo

# Step 1: generate
for w in "${WLEVELS[@]}"; do
    cfg="iupred_target_w${w}"
    out="$OUT_ROOT/$cfg"
    n_pdb=$(ls "$out/guided"/*.pdb 2>/dev/null | wc -l)
    if [ "$n_pdb" -ge 48 ]; then
        echo "[$(date -u +%FT%TZ)] [$DEVICE] $cfg already has $n_pdb PDBs; skipping generation"
        continue
    fi
    echo "[$(date -u +%FT%TZ)] [$DEVICE] starting $cfg generation"
    "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long \
        --steering_config "steering/config/sweep_iupred_target/${cfg}.yaml" \
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
echo "[$(date -u +%FT%TZ)] === iupred_target pipeline complete ==="
