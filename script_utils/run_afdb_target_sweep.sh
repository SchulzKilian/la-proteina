#!/usr/bin/env bash
# ALL-PROPERTIES -> AFDB-AVERAGE target steering, w-sweep.
#
# Every one of the 14 predictor properties is steered with direction=target and
# target_value = the predictor's own training mean (= the AFDB population average
# for that property). Because guide.py z-scores the target with the predictor's
# stats (target_z = (target - mean)/std), target_value = mean => target_z = 0:
# the objective is sum_i -w*(pred_z_i)^2, i.e. "pull every property to the
# AFDB centroid / make a maximally typical protein".
#
# Predictor: NOISE-AWARE multitask 5-fold ENSEMBLE (multitask_t1_noise_aware),
# denoised input (feed_z_t_directly=false, the best recipe per the lab notes).
# This is the same predictor + recipe as results/noise_aware_ensemble_sweep,
# so the seed-matched unguided codesign baseline anchors the deltas directly.
#
# Grid: w in {8,16,32} (gentle / standard-ceiling / aggressive-but-pre-crash;
# noise-aware codesign holds to ~w16, craters ~w48 per E114) x seeds 42-57 (16)
# x L in {300,400,500} x nsteps=400. ~45 s/protein on one L4 => ~2 h gen total.
#
# Usage:
#   nohup bash script_utils/run_afdb_target_sweep.sh > nohup_afdb_target_sweep.out 2>&1 &
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
export IUPRED3_DIR=${IUPRED3_DIR:-/home/ks2218/iupred3}

NA_DIR=laproteina_steerability/logs/multitask_t1_noise_aware/20260505_110348
CKPTS=()
for k in 0 1 2 3 4; do
    f="$(pwd)/${NA_DIR}/checkpoints/fold_${k}_best.pt"
    if [ ! -f "$f" ]; then echo "[ERR] missing $f"; exit 1; fi
    CKPTS+=("$f")
done
echo "[$(date -u +%FT%TZ)] noise-aware multitask ensemble = ${#CKPTS[@]} folds from $NA_DIR"

CFGDIR=steering/config/sweep_afdb_target
mkdir -p "$CFGDIR"

# --- build the all-property target objectives block once (ensemble-mean targets) ---
OBJ_BLOCK=$("$PY" - "${CKPTS[@]}" <<'PYEOF'
import sys, numpy as np, torch
from steering.registry import PROPERTY_NAMES
ckpts = sys.argv[1:]
means = []
for c in ckpts:
    d = torch.load(c, map_location="cpu", weights_only=False)
    means.append(np.array(d["stats_mean"], dtype=np.float64))
m = np.stack(means).mean(0)
assert len(m) == len(PROPERTY_NAMES), (len(m), len(PROPERTY_NAMES))
lines = []
for name, val in zip(PROPERTY_NAMES, m):
    lines.append(f"  - property: {name}")
    lines.append(f"    direction: target")
    lines.append(f"    target_value: {val:.8g}")
    lines.append(f"    weight: 1.0")
print("\n".join(lines))
PYEOF
)
echo "[$(date -u +%FT%TZ)] built target objectives for $(grep -c 'property:' <<<"$OBJ_BLOCK") properties"

for w in 16 32 64; do
    {
        echo "# ALL-PROPERTIES -> AFDB-AVERAGE target steering (auto-written by run_afdb_target_sweep.sh)."
        echo "# target_value per property = noise-aware ensemble training mean => target_z = 0."
        echo "steering:"
        echo "  enabled: true"
        echo "  checkpoint:"
        for f in "${CKPTS[@]}"; do echo "  - $f"; done
        echo "  objectives:"
        echo "$OBJ_BLOCK"
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
    } > "$CFGDIR/afdb_target_w${w}.yaml"
done
echo "[$(date -u +%FT%TZ)] wrote configs to $CFGDIR"

SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400 500"
NSTEPS=400
DEVICE=cuda:0
OUT_ROOT=results/afdb_target_sweep
mkdir -p "$OUT_ROOT"

for w in 16 32 64; do
    cfg="afdb_target_w${w}"; out="$OUT_ROOT/$cfg"
    n_pdb=$(ls "$out/guided"/*.pdb 2>/dev/null | wc -l)
    if [ "$n_pdb" -ge 48 ]; then echo "[$(date -u +%FT%TZ)] $cfg has $n_pdb PDBs; skip gen"; continue; fi
    echo "[$(date -u +%FT%TZ)] [$DEVICE] generating $cfg"
    "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long \
        --steering_config "$CFGDIR/${cfg}.yaml" \
        --lengths $LENGTHS --seeds $SEEDS --nsteps $NSTEPS --skip_unguided \
        --output_dir "$out" --device $DEVICE
done

echo "[$(date -u +%FT%TZ)] === audit (property + codesign + diversity) ==="
"$PY" script_utils/steering_cost_audit.py --tree "$OUT_ROOT" --evals property,codesign,diversity
echo "[$(date -u +%FT%TZ)] === AFDB-average target sweep complete ==="
