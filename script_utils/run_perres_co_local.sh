#!/bin/bash
# Local L4 driver for the per-residue CO throttle frontier. Resume-safe.
# One worker = one GPU + a list of cells. Each cell: <kind>:<w>
#   kind in {baseline, geometricres, ramares}; baseline = no-throttle (beta=0).
# Usage:
#   CUDA=0 CELLS="geometricres:16 geometricres:8" bash script_utils/run_perres_co_local.sh
#
# torch import is ~4s on the warm NFS cache; in-process ProteinMPNN (MPNN_INPROCESS=1)
# imports torch once per worker and reuses it for all generations + MPNN calls.
set -o pipefail
cd /home/ks2218/la-proteina

export PATH=$HOME/.conda/envs/laproteina_env/bin:$PATH
export MPNN_INPROCESS=1
export TANGO_EXE=${TANGO_EXE:-/home/ks2218/la-proteina/tango_x86_64_release}
PY=$HOME/.conda/envs/laproteina_env/bin/python

GPU="${CUDA:-0}"
export CUDA_VISIBLE_DEVICES=$GPU      # worker sees its GPU as cuda:0
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57}"
LENGTHS="${LENGTHS:-300 400}"
NSTEPS=400
ROOT=results/geom_lookahead_sweep
PRIORS=steering/throttle_priors/ca_pseudo_rama.pt
CFGDIR=steering/config/geometric_lookahead/perres_frontier
mkdir -p "$CFGDIR"

echo "[worker] gpu(phys)=$GPU  cells='$CELLS'  seeds='$SEEDS'  L='$LENGTHS'  nsteps=$NSTEPS"
ulimit -n 65536 2>/dev/null || true

beta_for () { case "$1" in geometric) echo 171.4809 ;; rama) echo 0.8721 ;; *) echo 1.0 ;; esac; }

write_cfg () {  # $1=proxy $2=w $3=beta $4=per_residue $5=cfgname -> path on stdout
  local proxy=$1 w=$2 beta=$3 per_res=$4 name=$5
  local f="$CFGDIR/${name}.yaml"
  {
    echo "steering:"
    echo "  method: geometric_lookahead"
    echo "  enabled: true"
    echo "  channel: bb_ca"
    echo "  mode: lookahead_proportional"
    echo "  proxy_type: ${proxy}"
    echo "  per_residue: ${per_res}"
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

gen_eval () {  # $1=cellname $2=cfgpath
  local cell=$1 cfg=$2
  echo "[$(date -u +%FT%TZ)] [gpu$GPU] gen $cell  (cfg=$cfg)"
  "$PY" -m steering.generate \
      --proteina_config inference_ucond_notri_long --steering_config "$cfg" \
      --lengths $LENGTHS --seeds $SEEDS --nsteps $NSTEPS --skip_unguided --resume \
      --output_dir "$ROOT/$cell" --device cuda:0
  echo "[$(date -u +%FT%TZ)] [gpu$GPU] scRMSD designability $cell"
  OUT_BASE=$ROOT "$PY" script_utils/run_scrmsd_steering.py \
      --cfgs "$cell" --seeds $SEEDS --lengths $LENGTHS
}

for spec in $CELLS; do
  kind=${spec%%:*}; w=${spec##*:}
  case "$kind" in
    baseline)
      cfg=$(write_cfg geometric "$w" 0.0 false "baseline_w${w}")
      gen_eval "contact_order_baseline_w${w}" "$cfg" ;;
    geometricres)
      b=$(beta_for geometric)
      cfg=$(write_cfg geometric "$w" "$b" true "geometricres_w${w}")
      gen_eval "contact_order_geometricres_w${w}" "$cfg" ;;
    ramares)
      b=$(beta_for rama)
      cfg=$(write_cfg rama "$w" "$b" true "ramares_w${w}")
      gen_eval "contact_order_ramares_w${w}" "$cfg" ;;
    *) echo "[skip] unknown cell spec: $spec" ;;
  esac
done
echo "[$(date -u +%FT%TZ)] [gpu$GPU] WORKER_DONE cells='$CELLS'"
