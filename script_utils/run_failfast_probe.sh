#!/usr/bin/env bash
# FAIL-FAST decisive probe for the per-residue CO throttle, using the LOCAL /dev/shm env
# (instant torch import). Only the two high-w cells where no-throttle collapses (~6%):
# geom w16 and rama w16, seeds 42-49, L300. Compared offline against the on-disk baseline
# curve (baseline_w8 n=32, w16 n=32). If the prior +72pp hint is real it shows at n=8.
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/dev/shm/lpenv/laproteina_env/bin/python
export MPNN_INPROCESS=1
export PATH=/dev/shm/lpenv/laproteina_env/bin:$PATH
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
export IUPRED3_DIR=${IUPRED3_DIR:-/home/ks2218/iupred3}

BETA_GEOM=171.4809
BETA_RAMA=0.8721
SEEDS="42 43 44 45 46 47 48 49"
LENGTHS="300"
NSTEPS=400
ROOT=results/geom_lookahead_sweep
PRIORS=steering/throttle_priors/ca_pseudo_rama.pt
CFGDIR=steering/config/geometric_lookahead/perres_frontier
mkdir -p "$CFGDIR"

write_cfg () {  # $1=proxy $2=w $3=beta -> path
  local proxy=$1 w=$2 beta=$3
  local f="$CFGDIR/${proxy}_w${w}_lookahead_proportional.yaml"
  {
    echo "steering:"
    echo "  method: geometric_lookahead"
    echo "  enabled: true"
    echo "  channel: bb_ca"
    echo "  mode: lookahead_proportional"
    echo "  proxy_type: ${proxy}"
    echo "  per_residue: true"
    echo "  f_map: exp"
    echo "  beta: ${beta}"
    echo "  objectives: [{property: contact_order, direction: minimize, weight: 1.0}]"
    echo "  schedule: {type: linear_ramp, w_max: ${w}.0, t_start: 0.3, t_end: 0.8}"
    if [ "$proxy" = "rama" ]; then
      echo "  proxy: {rama_priors_path: ${PRIORS}}"
    else
      echo "  proxy: {bond_target_nm: 0.38, clash_radius_nm: 0.40, lambda_clash: 1.0}"
    fi
    echo "  gradient_norm: unit"
    echo "  gradient_clip: 0.0"
    echo "  log_diagnostics: true"
  } > "$f"
  echo "$f"
}

probe () {  # $1=proxy $2=beta $3=gpu
  local proxy=$1 beta=$2 gpu=$3
  local cell="contact_order_${proxy}res_w16"
  local cfg; cfg=$(write_cfg "$proxy" 16 "$beta")
  echo "[$(date -u +%FT%TZ)] [GPU$gpu] gen $cell seeds=$SEEDS L=$LENGTHS"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -m steering.generate \
      --proteina_config inference_ucond_notri_long --steering_config "$cfg" \
      --lengths $LENGTHS --seeds $SEEDS --nsteps $NSTEPS --skip_unguided \
      --output_dir "$ROOT/$cell" --device cuda:0
  echo "[$(date -u +%FT%TZ)] [GPU$gpu] scRMSD $cell"
  CUDA_VISIBLE_DEVICES=$gpu OUT_BASE=$ROOT "$PY" script_utils/run_scrmsd_steering.py \
      --cfgs "$cell" --seeds $SEEDS --lengths $LENGTHS
  echo "[$(date -u +%FT%TZ)] [GPU$gpu] $cell DONE"
}

echo "[$(date -u +%FT%TZ)] === FAIL-FAST PROBE (local env) ==="
probe geometric "$BETA_GEOM" 0 > nohup_probe_geom.log 2>&1 &
probe rama      "$BETA_RAMA" 3 > nohup_probe_rama.log 2>&1 &
wait
echo "[$(date -u +%FT%TZ)] === PROBE_DONE -> verdict ==="
"$PY" script_utils/failfast_verdict.py 300
