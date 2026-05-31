#!/usr/bin/env bash
# E110 decisive test: iupred3_fraction_disordered TARGET=0.123 steering driven by the
# CONCEPT-BOTTLENECK predictor (single best CBM fold_2) instead of the NA-v1 ensemble.
# Recipe IDENTICAL to E109's w32 cell (inference_ucond_notri_long, linear_ramp 0.3/0.8/0.9,
# unit grad-norm, denoised, nsteps=400, 16 seeds x L{300,400,500} = 48 PDBs), so the
# per-sample predicted<->real iupred correlation is directly comparable to E109 (NA-v1: r=0.43 at w32).
#
# Usage: nohup bash script_utils/run_iupred_cbm_probe.sh > nohup_iupred_cbm.out 2>&1 &
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
OUT_ROOT=results/iupred_target_cbm
cfg=iupred_target_cbm_w32
out="$OUT_ROOT/$cfg"
mkdir -p "$OUT_ROOT"

echo "[$(date -u +%FT%TZ)] === CBM-driven iupred target=0.123 w32 probe ==="
n_pdb=$(ls "$out/guided"/*.pdb 2>/dev/null | wc -l)
if [ "$n_pdb" -ge 48 ]; then
    echo "[$(date -u +%FT%TZ)] $cfg already has $n_pdb PDBs; skipping generation"
else
    echo "[$(date -u +%FT%TZ)] [$DEVICE] generating $cfg (48 PDBs, nsteps=$NSTEPS)"
    "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long \
        --steering_config "steering/config/sweep_iupred_target_cbm/${cfg}.yaml" \
        --lengths $LENGTHS \
        --seeds $SEEDS \
        --nsteps $NSTEPS \
        --skip_unguided \
        --output_dir "$out" \
        --device $DEVICE
    echo "[$(date -u +%FT%TZ)] generation done"
fi

echo "[$(date -u +%FT%TZ)] === developability (real IUPred3) + codesign on guided PDBs ==="
"$PY" script_utils/steering_cost_audit.py --tree "$OUT_ROOT" --evals property,codesign

echo "[$(date -u +%FT%TZ)] === CBM iupred probe complete ==="
