#!/usr/bin/env bash
# TEMPORARY self-contained A100 driver for the throttle sweep.
# Portable: relative ckpt paths (no absolute /home/... ), assumes you've activated
# the env (`python` on PATH) and the 3 ckpts are in place (see header of transfer cmd).
#   - LD3/AE2 referenced relatively by inference_ucond_notri_long -> checkpoints_laproteina/
#   - CBM fold_2 referenced relatively below
# Generation batched at BATCH_SIZE (default 16); ESMFold in the audit stays sequential.
#
# Usage (from repo root, env activated):
#   CUDA_VISIBLE_DEVICES=<gpu> nohup bash script_utils/run_throttle_sweep_a100.sh > nohup_throttle_sweep.log 2>&1 &
#   # smaller card: BATCH_SIZE=8 CUDA_VISIBLE_DEVICES=<gpu> nohup bash script_utils/run_throttle_sweep_a100.sh ...
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root, regardless of where it's launched from
PY=${PY:-python}          # assumes laproteina_env is activated
export TANGO_EXE=${TANGO_EXE:-$(pwd)/tango_x86_64_release}
export IUPRED3_DIR=${IUPRED3_DIR:-$HOME/iupred3}

# --- real ckpt paths, repo-relative (portable) ---
CKPT=laproteina_steerability/logs/multitask_cbm/20260531_121832/checkpoints/fold_2_best.pt
PRIORS=steering/throttle_priors/priors.pt
for f in "$CKPT" "$PRIORS" checkpoints_laproteina/LD3_ucond_notri_800.ckpt checkpoints_laproteina/AE2_ucond_800.ckpt; do
    [ -f "$f" ] || { echo "[ERR] missing required file: $f"; exit 1; }
done

CFGDIR=steering/config/sweep_throttle_full
OUT_ROOT=results/throttle_sweep
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400 500"
NSTEPS=400
DEVICE=cuda:0
BATCH_SIZE=${BATCH_SIZE:-16}
mkdir -p "$CFGDIR" "$OUT_ROOT"

declare -A ARM_THROTTLE=( [nothrottle]="none 0.0" [rama]="rama 0.25" [aaprior]="aa_prior 40.0" )

write_cfg() {  # $1=arm $2=w
    local arm=$1 w=$2 ttype tbeta
    read -r ttype tbeta <<< "${ARM_THROTTLE[$arm]}"
    {
        echo "# tango_min MINIMIZE, single CBM fold_2, w${w}, throttle=${arm} (auto-written)."
        echo "steering:"
        echo "  enabled: true"
        echo "  checkpoint:"
        echo "  - $CKPT"
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
        if [ "$ttype" != "none" ]; then
            echo "    beta: ${tbeta}"
            echo "    priors_path: ${PRIORS}"
        fi
    } > "$CFGDIR/${arm}_w${w}.yaml"
}

for w in 128 64; do
  for arm in nothrottle rama aaprior; do
    write_cfg "$arm" "$w"
    out="$OUT_ROOT/$arm/tango_min_w${w}"
    n_pdb=$(ls "$out/guided"/*.pdb 2>/dev/null | wc -l)
    if [ "$n_pdb" -ge 48 ]; then echo "[$(date -u +%FT%TZ)] $arm w$w has $n_pdb PDBs; skip gen"; continue; fi
    echo "[$(date -u +%FT%TZ)] [$DEVICE bs=$BATCH_SIZE] generating arm=$arm w=$w"
    "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long \
        --steering_config "$CFGDIR/${arm}_w${w}.yaml" \
        --lengths $LENGTHS --seeds $SEEDS --nsteps $NSTEPS --skip_unguided \
        --batch_size $BATCH_SIZE \
        --output_dir "$out" --device $DEVICE
  done
done

echo "[$(date -u +%FT%TZ)] === audit (codesign + real TANGO + diversity), per arm ==="
for arm in nothrottle rama aaprior; do
    echo "[$(date -u +%FT%TZ)] --- audit arm=$arm ---"
    "$PY" script_utils/steering_cost_audit.py --tree "$OUT_ROOT/$arm" --evals property,codesign,diversity
done
echo "[$(date -u +%FT%TZ)] === throttle full sweep complete ==="
