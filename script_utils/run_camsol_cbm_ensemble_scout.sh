#!/usr/bin/env bash
# camsol_max Pareto driven by the NOISE-AWARE CBM as a 5-FOLD ENSEMBLE.
# Resolves the 5 fold ckpts at runtime (newest logs/multitask_cbm_noise_aware/<ts>/), writes the
# ensemble steering configs, then runs the same w{16,32,64,128} x 16-seed x L{300,400,500} sweep
# as the single-fold camsol run + E066, so the SWI-vs-codesign frontier is directly comparable:
#   single non-NA CBM (results/camsol_cbm_scout) vs NA ensemble CBM (here) vs NA-v1 (E066).
#
# Usage (chained after the noise-aware training finishes):
#   nohup bash script_utils/run_camsol_cbm_ensemble_scout.sh > nohup_camsol_cbm_na_ens.out 2>&1 &
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
export IUPRED3_DIR=${IUPRED3_DIR:-/home/ks2218/iupred3}

# --- resolve the newest noise-aware CBM run + its 5 fold ckpts ---
NA_DIR=$(ls -td laproteina_steerability/logs/multitask_cbm_noise_aware/*/ 2>/dev/null | head -1)
if [ -z "$NA_DIR" ]; then echo "[ERR] no noise-aware CBM run dir found"; exit 1; fi
CKPTS=()
for k in 0 1 2 3 4; do
    f="$(pwd)/${NA_DIR}checkpoints/fold_${k}_best.pt"
    if [ ! -f "$f" ]; then echo "[ERR] missing $f — training incomplete?"; exit 1; fi
    CKPTS+=("$f")
done
echo "[$(date -u +%FT%TZ)] noise-aware CBM ensemble = ${#CKPTS[@]} folds from $NA_DIR"

# --- write the 5-fold-ensemble steering configs ---
CFGDIR=steering/config/sweep_camsol_cbm_na_ensemble
mkdir -p "$CFGDIR"
for w in 16 32 64 128; do
    {
        echo "# camsol_max MAXIMIZE, NOISE-AWARE CBM 5-fold ENSEMBLE (auto-written by run_camsol_cbm_ensemble_scout.sh)."
        echo "steering:"
        echo "  enabled: true"
        echo "  checkpoint:"
        for f in "${CKPTS[@]}"; do echo "  - $f"; done
        echo "  objectives:"
        echo "  - property: camsol_intrinsic"
        echo "    direction: maximize"
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
    } > "$CFGDIR/camsol_max_w${w}.yaml"
done
echo "[$(date -u +%FT%TZ)] wrote ensemble configs to $CFGDIR"

SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400 500"
NSTEPS=400
DEVICE=cuda:0
OUT_ROOT=results/camsol_cbm_na_ensemble_scout
mkdir -p "$OUT_ROOT"

for w in 16 32 64 128; do
    cfg="camsol_max_w${w}"; out="$OUT_ROOT/$cfg"
    n_pdb=$(ls "$out/guided"/*.pdb 2>/dev/null | wc -l)
    if [ "$n_pdb" -ge 48 ]; then echo "[$(date -u +%FT%TZ)] $cfg has $n_pdb PDBs; skip gen"; continue; fi
    echo "[$(date -u +%FT%TZ)] [$DEVICE] generating $cfg (ensemble)"
    "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long \
        --steering_config "$CFGDIR/${cfg}.yaml" \
        --lengths $LENGTHS --seeds $SEEDS --nsteps $NSTEPS --skip_unguided \
        --output_dir "$out" --device $DEVICE
done

echo "[$(date -u +%FT%TZ)] === audit (SWI + codesign + diversity) ==="
"$PY" script_utils/steering_cost_audit.py --tree "$OUT_ROOT" --evals property,codesign,diversity
echo "[$(date -u +%FT%TZ)] === noise-aware CBM ensemble camsol scout complete ==="
