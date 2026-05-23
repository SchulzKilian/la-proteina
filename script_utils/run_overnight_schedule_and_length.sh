#!/usr/bin/env bash
# Overnight chained pipeline:
#  (1) Schedule reshape at fixed tango_min w=32 — 3 schedule variants
#      (early / late / wide) × 16 seeds × 3 lengths = 144 PDBs
#  (2) Length-extension Pareto for tango_min at L=400 and L=500 only
#      (skip L=300, which we have plenty of) — 16 NEW seeds (58-73) ×
#      4 w-levels (16/32/64/128) × 2 lengths = 128 PDBs
#  Per-phase audit (property + AA + codesign + diversity) on each tree.
set -uo pipefail
cd /home/ks2218/la-proteina

PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
SEEDS_ORIG="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
SEEDS_NEW="58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73"
NSTEPS=400
DEVICE=cuda:0

# ---- PHASE 1: schedule reshape ----
OUT_ROOT_SCHED=results/tango_min_schedule_scout
mkdir -p "$OUT_ROOT_SCHED"
echo "[$(date)] === PHASE 1: schedule reshape (tango_min @ w=32, 3 variants) ==="
for variant in early late wide; do
    cfg="tango_min_${variant}_w32"
    out="$OUT_ROOT_SCHED/$cfg"
    echo "[$(date)] [$DEVICE] starting $cfg"
    "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long \
        --steering_config "steering/config/sweep_tango_min_schedule/${cfg}.yaml" \
        --lengths 300 400 500 \
        --seeds $SEEDS_ORIG \
        --nsteps $NSTEPS \
        --skip_unguided \
        --output_dir "$out" \
        --device $DEVICE
    echo "[$(date)] [$DEVICE] finished $cfg"
done
echo "[$(date)] === PHASE 1 generation done; running audit ==="
"$PY" script_utils/steering_cost_audit.py \
    --tree "$OUT_ROOT_SCHED" \
    --evals property,aa,codesign,diversity
echo "[$(date)] === PHASE 1 complete ==="
echo

# ---- PHASE 2: length-extension (tango_min at L=400 and L=500, new seeds) ----
OUT_ROOT_LEN=results/tango_min_length_ext
mkdir -p "$OUT_ROOT_LEN"
echo "[$(date)] === PHASE 2: length-extension Pareto (tango_min, L=400+500, seeds 58-73) ==="
for w in 16 32 64 128; do
    cfg="tango_min_w${w}"
    out="$OUT_ROOT_LEN/$cfg"
    # w=16 lives in sweep_noise_aware_ensemble; w∈{32,64,128} in sweep_noise_aware_high_w
    if [[ "$w" == "16" ]]; then
        cfg_path="steering/config/sweep_noise_aware_ensemble/tango_min_w${w}.yaml"
    else
        cfg_path="steering/config/sweep_noise_aware_high_w/tango_min_w${w}.yaml"
    fi
    echo "[$(date)] [$DEVICE] starting length-ext $cfg (config: $cfg_path)"
    "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long \
        --steering_config "$cfg_path" \
        --lengths 400 500 \
        --seeds $SEEDS_NEW \
        --nsteps $NSTEPS \
        --skip_unguided \
        --output_dir "$out" \
        --device $DEVICE
    echo "[$(date)] [$DEVICE] finished length-ext $cfg"
done
echo "[$(date)] === PHASE 2 generation done; running audit ==="
"$PY" script_utils/steering_cost_audit.py \
    --tree "$OUT_ROOT_LEN" \
    --evals property,aa,codesign,diversity \
    --seeds 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 \
    --lengths 400 500
echo "[$(date)] === PHASE 2 complete ==="
echo
echo "[$(date)] === OVERNIGHT PIPELINE COMPLETE ==="
