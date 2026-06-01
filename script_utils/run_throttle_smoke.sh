#!/usr/bin/env bash
# SMOKE: confirm the steering throttle (BLOCKER) fires and damps as designed.
# tango_min, single CBM fold_2, w128, nsteps=400. Three variants:
#   nothrottle (baseline) | rama (per-residue veto) | aaprior (per-protein composition guard).
# N=2 seeds x L=300 only. Goal = inspect per-step throttle diagnostics (s, dP), not a frontier.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=5 nohup bash script_utils/run_throttle_smoke.sh > nohup_throttle_smoke.log 2>&1 &
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
export IUPRED3_DIR=${IUPRED3_DIR:-/home/ks2218/iupred3}

CFGDIR=steering/config/sweep_throttle_smoke
OUT_ROOT=results/throttle_smoke
LENGTHS=300
SEEDS="42 43"
NSTEPS=400
DEVICE=cuda:0          # CUDA_VISIBLE_DEVICES pins the physical GPU; torch sees cuda:0
mkdir -p "$OUT_ROOT"

for cfg in tango_min_w128_nothrottle tango_min_w128_rama tango_min_w128_aaprior; do
    out="$OUT_ROOT/$cfg"
    n_pdb=$(ls "$out/guided"/*.pdb 2>/dev/null | wc -l)
    if [ "$n_pdb" -ge 2 ]; then echo "[$(date -u +%FT%TZ)] $cfg has $n_pdb PDBs; skip"; continue; fi
    echo "[$(date -u +%FT%TZ)] [$DEVICE] generating $cfg"
    "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long \
        --steering_config "$CFGDIR/${cfg}.yaml" \
        --lengths $LENGTHS --seeds $SEEDS --nsteps $NSTEPS --skip_unguided \
        --output_dir "$out" --device $DEVICE
done

echo "[$(date -u +%FT%TZ)] === throttle smoke generation complete ==="
echo "Diagnostics: $OUT_ROOT/*/diagnostics/*.json  (look for the 'throttle' sub-dict per step)"
