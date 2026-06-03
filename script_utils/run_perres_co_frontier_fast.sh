#!/usr/bin/env bash
# FAST relaunch of the per-residue CO frontier. Two fixes vs run_perres_co_frontier.sh:
#   (1) Runs Python from a LOCAL copy of the conda env (/tmp/laproteina_env) so the
#       per-protein ProteinMPNN subprocess no longer re-imports torch over slow NFS
#       (~9.5 min/protein -> seconds). PYTHON_EXEC is exported so run_proteinmpnn picks it up.
#   (2) Fans the 8 cells (geom+rama x w{4,8,16,32}) across 4 GPUs instead of 2.
# Betas reuse the already-completed Stage-1 calibration (geom=171.48, rama=0.8721) -> no recalib.
# gen skips cells that already have 32 pdb; scRMSD is resume-safe (skips done protein_ids).
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
export MPNN_INPROCESS=1                     # <- in-process ProteinMPNN: torch imported once per worker, not per protein
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
export IUPRED3_DIR=${IUPRED3_DIR:-/home/ks2218/iupred3}

BETA_GEOM=171.4809
BETA_RAMA=0.8721
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400"
NSTEPS=400
ROOT=results/geom_lookahead_sweep
PRIORS=steering/throttle_priors/ca_pseudo_rama.pt
CFGDIR=steering/config/geometric_lookahead/perres_frontier
mkdir -p "$CFGDIR"

write_cfg () {  # $1=proxy $2=w $3=beta $4=mode -> path on stdout
  local proxy=$1 w=$2 beta=$3 mode=$4
  local f="$CFGDIR/${proxy}_w${w}_${mode}.yaml"
  {
    echo "steering:"
    echo "  method: geometric_lookahead"
    echo "  enabled: true"
    echo "  channel: bb_ca"
    echo "  mode: ${mode}"
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

run_cell () {  # $1=proxy $2=w $3=beta $4=gpu
  local proxy=$1 w=$2 beta=$3 gpu=$4
  local cell="contact_order_${proxy}res_w${w}"
  local n_pdb; n_pdb=$(ls "$ROOT/$cell/guided"/*.pdb 2>/dev/null | wc -l)
  if [ "$n_pdb" -lt 32 ]; then
    local cfg; cfg=$(write_cfg "$proxy" "$w" "$beta" lookahead_proportional)
    echo "[$(date -u +%FT%TZ)] [GPU$gpu] gen $cell (beta=$beta)"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long --steering_config "$cfg" \
        --lengths $LENGTHS --seeds $SEEDS --nsteps $NSTEPS --skip_unguided \
        --output_dir "$ROOT/$cell" --device cuda:0
  else
    echo "[$(date -u +%FT%TZ)] $cell has $n_pdb pdbs; skip gen"
  fi
  echo "[$(date -u +%FT%TZ)] [GPU$gpu] scRMSD designability $cell"
  CUDA_VISIBLE_DEVICES=$gpu OUT_BASE=$ROOT "$PY" script_utils/run_scrmsd_steering.py \
      --cfgs "$cell" --seeds $SEEDS --lengths $LENGTHS
}

# GPU assignment: 4 GPUs, each handles a sequential list of (proxy,w) cells.
gpu_worker () {  # $1=gpu $2=proxy $3="w w ..." $4=beta
  local gpu=$1 proxy=$2 ws=$3 beta=$4
  for w in $ws; do run_cell "$proxy" "$w" "$beta" "$gpu"; done
  echo "[$(date -u +%FT%TZ)] === worker gpu$gpu ($proxy: $ws) DONE ==="
}

echo "[$(date -u +%FT%TZ)] === FAST per-residue CO frontier: 4-GPU fan-out ==="
echo "[$(date -u +%FT%TZ)] python=$PY  MPNN_INPROCESS=$MPNN_INPROCESS"
gpu_worker 0 geometric "4 8"  "$BETA_GEOM" > nohup_perres_g0.log 2>&1 &
gpu_worker 1 geometric "16 32" "$BETA_GEOM" > nohup_perres_g1.log 2>&1 &
gpu_worker 3 rama      "4 8"  "$BETA_RAMA" > nohup_perres_g3.log 2>&1 &
gpu_worker 5 rama      "16 32" "$BETA_RAMA" > nohup_perres_g5.log 2>&1 &
wait

# baseline top-up: bring no-throttle contact_order w4/w32 to 16 seeds for a clean reference.
echo "[$(date -u +%FT%TZ)] === baseline top-up (w4,w32 -> 16 seeds) ==="
CUDA_VISIBLE_DEVICES=0 "$PY" -m steering.run_geom_lookahead_sweep --device cuda:0 \
    --targets contact_order --modes baseline --lambdas 4 32 --seeds 48 49 50 51 52 53 54 55 56 57 >/dev/null 2>&1 || true
for w in 4 32; do
  CUDA_VISIBLE_DEVICES=0 OUT_BASE=$ROOT "$PY" script_utils/run_scrmsd_steering.py \
      --cfgs "contact_order_baseline_w${w}" --seeds 48 49 50 51 52 53 54 55 56 57 --lengths $LENGTHS || true
done

echo "[$(date -u +%FT%TZ)] === PERRES_FRONTIER_FAST_DONE ==="
