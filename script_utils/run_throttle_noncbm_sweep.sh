#!/usr/bin/env bash
# Throttle sweep with NON-CBM steering: steer with the 5-fold multitask_t1_noise_aware
# ensemble (the same predictor as the existing no-throttle baseline in
# results/noise_aware_high_w_scout/), throttle reads a SEPARATE CBM g1. Tests whether
# the rama / aa_prior blocker RESCUES codesignability on the Goodhart-prone non-CBM
# predictor at high w. No-throttle arm is NOT regenerated (baseline already exists).
#
# Arms: rama, aaprior. w{64,128}. 16 seeds x L{300,400,500}. nsteps=400. B=1 (L4).
# betas come from the non-CBM Stage-1 calibration (set RAMA_BETA / AA_BETA below).
#
# Usage:
#   RAMA_BETA=0.25 AA_BETA=40 CUDA_VISIBLE_DEVICES=5 nohup bash script_utils/run_throttle_noncbm_sweep.sh > nohup_throttle_noncbm.log 2>&1 &
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
export IUPRED3_DIR=${IUPRED3_DIR:-/home/ks2218/iupred3}

NONCBM_DIR=/home/ks2218/la-proteina/laproteina_steerability/logs/multitask_t1_noise_aware/20260505_110348/checkpoints
CBM_G1=/home/ks2218/la-proteina/laproteina_steerability/logs/multitask_cbm/20260531_121832/checkpoints/fold_2_best.pt
PRIORS=steering/throttle_priors/priors.pt
CFGDIR=steering/config/sweep_throttle_noncbm
OUT_ROOT=results/throttle_noncbm
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400 500"
NSTEPS=400
DEVICE=cuda:0
BATCH_SIZE=${BATCH_SIZE:-1}
RAMA_BETA=${RAMA_BETA:-0.25}
AA_BETA=${AA_BETA:-40.0}
mkdir -p "$CFGDIR" "$OUT_ROOT"

declare -A ARM_THROTTLE=( [rama]="rama $RAMA_BETA" [aaprior]="aa_prior $AA_BETA" )

write_cfg() {  # $1=arm $2=w
    local arm=$1 w=$2 ttype tbeta
    read -r ttype tbeta <<< "${ARM_THROTTLE[$arm]}"
    {
        echo "# tango_min, NON-CBM 5-fold NA ensemble steer + CBM-g1 ${arm} throttle, w${w} (auto)."
        echo "steering:"
        echo "  enabled: true"
        echo "  checkpoint:"
        for k in 0 1 2 3 4; do echo "  - $NONCBM_DIR/fold_${k}_best.pt"; done
        echo "  objectives:"
        echo "  - property: tango"
        echo "    direction: minimize"
        echo "    weight: 1.0"
        echo "  schedule:"
        echo "    type: linear_ramp"
        echo "    w_max: ${w}.0"
        echo "    t_start: 0.3"
        echo "    t_end: 0.8"
        echo "    t_stop: 0.9"
        echo "  gradient_norm: unit"
        echo "  gradient_clip: 10.0"
        echo "  channel: local_latents"
        echo "  log_diagnostics: true"
        echo "  throttle:"
        echo "    type: ${ttype}"
        echo "    beta: ${tbeta}"
        echo "    g1_checkpoint: $CBM_G1"
        echo "    priors_path: $PRIORS"
    } > "$CFGDIR/${arm}_w${w}.yaml"
}

for w in 128 64; do
  for arm in rama aaprior; do
    write_cfg "$arm" "$w"
    out="$OUT_ROOT/$arm/tango_min_w${w}"
    n_pdb=$(ls "$out/guided"/*.pdb 2>/dev/null | wc -l)
    if [ "$n_pdb" -ge 48 ]; then echo "[$(date -u +%FT%TZ)] $arm w$w has $n_pdb PDBs; skip gen"; continue; fi
    echo "[$(date -u +%FT%TZ)] [$DEVICE bs=$BATCH_SIZE] gen arm=$arm w=$w (beta=${ARM_THROTTLE[$arm]##* })"
    "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long \
        --steering_config "$CFGDIR/${arm}_w${w}.yaml" \
        --lengths $LENGTHS --seeds $SEEDS --nsteps $NSTEPS --skip_unguided \
        --batch_size $BATCH_SIZE \
        --output_dir "$out" --device $DEVICE
  done
done

echo "[$(date -u +%FT%TZ)] === audit rama + aaprior (codesign cached by GPU6 watcher; property+diversity here) ==="
for arm in rama aaprior; do
    echo "[$(date -u +%FT%TZ)] --- audit arm=$arm ---"
    "$PY" script_utils/steering_cost_audit.py --tree "$OUT_ROOT/$arm" --evals property,codesign,diversity
done
echo "[$(date -u +%FT%TZ)] === non-CBM throttle sweep complete ==="
