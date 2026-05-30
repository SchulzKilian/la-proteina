#!/usr/bin/env bash
# Single-objective TARGET steering sweep: net_charge -> target_value = -5.0 (natural).
#
# First real test of direction: target in the repo (all prior sweeps used
# minimize/maximize). Question: with unit-normalized gradients, does the target
# objective settle the net charge AT the natural target (-5) or oscillate/over-
# shoot? E103 (maximize) blew past natural to +20..+103; target should self-
# correct direction once it crosses -5.
#
# Recipe IDENTICAL to E103 (net_charge_max) except the objective direction, so
# the two are directly comparable:
#   NA-v1 5-fold ensemble predictor, inference_ucond_notri_long,
#   linear_ramp (t_start=0.3, t_end=0.8, t_stop=0.9), unit grad-norm,
#   denoised input (feed_z_t_directly unset = False = best recipe), nsteps=400,
#   16 seeds x L in {300,400,500}, w in {8,16,24,32} = 192 PDBs.
# w grid is the GENTLE regime: target self-corrects direction, so lower w should
# settle closer to -5 with less codesign cost.
#
# Audit anchored against the n=30 paired unsteered baseline (net_charge -20.4).
#
# Usage:
#   nohup bash script_utils/run_net_charge_target_pipeline.sh \
#       > nohup_net_charge_target.out 2>&1 &
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
OUT_ROOT=results/net_charge_target_sweep
WLEVELS=(8 16 24 32)

mkdir -p "$OUT_ROOT"
echo "[$(date -u +%FT%TZ)] === net_charge TARGET=-5 sweep starting ==="
echo "4 w-levels x 48 PDBs each = 192 generations, single-objective target steering."
echo

# Step 1: generate
for w in "${WLEVELS[@]}"; do
    cfg="net_charge_target_w${w}"
    out="$OUT_ROOT/$cfg"
    n_pdb=$(ls "$out/guided"/*.pdb 2>/dev/null | wc -l)
    if [ "$n_pdb" -ge 48 ]; then
        echo "[$(date -u +%FT%TZ)] [$DEVICE] $cfg already has $n_pdb PDBs; skipping generation"
        continue
    fi
    echo "[$(date -u +%FT%TZ)] [$DEVICE] starting $cfg generation"
    "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long \
        --steering_config "steering/config/sweep_net_charge_target/${cfg}.yaml" \
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
echo "[$(date -u +%FT%TZ)] === net_charge_target pipeline complete ==="
