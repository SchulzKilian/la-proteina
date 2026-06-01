#!/usr/bin/env bash
# E110 dose-response: iupred3_fraction_disordered TARGET=0.123 steering driven by the
# CONCEPT-BOTTLENECK predictor (single best CBM fold_2), swept over w in {8,16,24} to
# complete the curve (w32 already done in results/iupred_target_cbm/iupred_target_cbm_w32).
# Goal: show the CBM's per-sample predicted<->real iupred correlation stays COUPLED across w,
# where the NA-v1 latent->property shortcut decoupled (E109: r 0.84->0.43 as w 8->32).
# Recipe IDENTICAL to the w32 cell (inference_ucond_notri_long, linear_ramp 0.3/0.8/0.9,
# unit grad-norm, denoised, nsteps=400, 16 seeds x L{300,400,500} = 48 PDBs per w).
#
# Usage: CUDA_VISIBLE_DEVICES=<g> nohup bash script_utils/run_iupred_cbm_dose_response.sh > nohup_iupred_cbm_dose.out 2>&1 &
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
export IUPRED3_DIR=${IUPRED3_DIR:-/home/ks2218/iupred3}

SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400 500"
NSTEPS=400
DEVICE=cuda:0                     # CUDA_VISIBLE_DEVICES pins the physical GPU; torch sees it as cuda:0
OUT_ROOT=results/iupred_target_cbm
mkdir -p "$OUT_ROOT"

for w in 8 16 24; do
    cfg=iupred_target_cbm_w${w}
    out="$OUT_ROOT/$cfg"
    echo "[$(date -u +%FT%TZ)] ===== CBM iupred target=0.123 w${w} ====="
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
        echo "[$(date -u +%FT%TZ)] generation done for $cfg"
    fi
done

echo "[$(date -u +%FT%TZ)] === developability (real IUPred3) + codesign across full tree (w8/16/24/32) ==="
"$PY" script_utils/steering_cost_audit.py --tree "$OUT_ROOT" --evals property,codesign

echo "[$(date -u +%FT%TZ)] === CBM iupred dose-response complete ==="
