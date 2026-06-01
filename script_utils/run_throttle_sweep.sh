#!/usr/bin/env bash
# FULL throttle sweep: does the CBM-g1 latent throttle preserve codesignability at
# high steering w (vs no-throttle), while keeping the tango effect?
#
# Objective: tango_min, single CBM fold_2. Three arms x w{64,128} x 16 seeds x L{300,400,500}.
#   nothrottle (baseline) | rama (per-residue veto, beta=0.25) | aaprior (per-protein, beta=40)
# Each arm in its OWN subtree with cells named tango_min_w{N} so steering_cost_audit detects
# direction=tango_min -> real column tango_total, codesign anchor from the seed-matched
# unsteered baseline CSV. nsteps=400. Resumable (skips cells with >=48 PDBs). Single GPU.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=5 nohup bash script_utils/run_throttle_sweep.sh > nohup_throttle_sweep.log 2>&1 &
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
export IUPRED3_DIR=${IUPRED3_DIR:-/home/ks2218/iupred3}

CKPT=/home/ks2218/la-proteina/laproteina_steerability/logs/multitask_cbm/20260531_121832/checkpoints/fold_2_best.pt
PRIORS=steering/throttle_priors/priors.pt
CFGDIR=steering/config/sweep_throttle_full
OUT_ROOT=results/throttle_sweep
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400 500"
NSTEPS=400
DEVICE=cuda:0
# Native nsamples batching for generation throughput (1 = original per-seed path).
# 16 seeds/length -> one forward per (arm,w,L). Safe on A100-80/40 at L<=500;
# drop to ~8 on a 24GB card. ESMFold in the audit stays sequential.
BATCH_SIZE=${BATCH_SIZE:-16}
mkdir -p "$CFGDIR" "$OUT_ROOT"

# arm -> "throttle_type beta"
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

# w128 first (most informative), then w64. nothrottle first per w (shared baseline).
for w in 128 64; do
  for arm in nothrottle rama aaprior; do
    write_cfg "$arm" "$w"
    out="$OUT_ROOT/$arm/tango_min_w${w}"
    n_pdb=$(ls "$out/guided"/*.pdb 2>/dev/null | wc -l)
    if [ "$n_pdb" -ge 48 ]; then echo "[$(date -u +%FT%TZ)] $arm w$w has $n_pdb PDBs; skip gen"; continue; fi
    echo "[$(date -u +%FT%TZ)] [$DEVICE] generating arm=$arm w=$w"
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
