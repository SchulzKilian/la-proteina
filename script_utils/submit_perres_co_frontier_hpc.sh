#!/bin/bash
#SBATCH -J perres_co
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_perres_co_%j.out
#
# Per-residue look-ahead THROTTLE frontier on contact_order (minimize), on Cambridge HPC.
# Two per-residue proxies: geometric (bond/clash) and Cα pseudo-rama. Grid:
#   proxy{geometric,rama} x w{4,8,16,32} x seeds{42..57} x L{300,400}, nsteps=400.
# 8-seq MPNN designability (scRMSD_ca_min<2A) per cell. Resume-safe (skips done pdbs/ids),
# so --requeue is fine. Closed-form geometric steering -> NO learned predictor needed;
# only the standard LD3+AE2 checkpoints + the rama priors file (both in the repo).
#
# PREREQUISITES on HPC (repo root = /home/ks2218/la-proteina):
#   - this repo, pushed from the L4 box, incl. steering/{throttle.py,guide_geometric.py,
#     geometry.py}, steering/throttle_priors/ca_pseudo_rama.pt, and the configs/*.
#   - checkpoints_laproteina/LD3_ucond_notri_800.ckpt and AE2_ucond_800.ckpt present
#     (these are what inference_ucond_notri_long.yaml points at).
#
# Submit:   sbatch script_utils/submit_perres_co_frontier_hpc.sh
# To halve wall-clock, submit twice with PROXIES=geometric and PROXIES=rama:
#   sbatch --export=ALL,PROXIES=geometric script_utils/submit_perres_co_frontier_hpc.sh
#   sbatch --export=ALL,PROXIES=rama      script_utils/submit_perres_co_frontier_hpc.sh
set -o pipefail   # NOT set -e, and NOT -u: sourcing .bashrc references unset $BASHRCSOURCED
                  # and `set -u` would abort the whole script there. Overridable vars use ${V:-default}.

cd /home/ks2218/la-proteina

# --- env (PATH prepend, NOT conda activate; canonical env on /home, never /rds) ---
source $HOME/.bashrc
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
export IUPRED3_DIR=${IUPRED3_DIR:-/home/ks2218/iupred3}
export MPNN_INPROCESS=1     # in-process ProteinMPNN (one torch import, reused); harmless if unsupported
PY=python
ulimit -n 65536 2>/dev/null || true
echo "node=$(hostname) python=$(which python) gpu=${CUDA_VISIBLE_DEVICES:-unset}"

# --- pre-flight: verify the long-generation checkpoints exist (paths from
#     configs/inference_ucond_notri_long.yaml). Fail fast with a clear message. ---
LD_CKPT=./checkpoints_laproteina/LD3_ucond_notri_800.ckpt   # latent flow (~2.8G)
AE_CKPT=./checkpoints_laproteina/AE2_ucond_800.ckpt         # autoencoder (~3.9G)
for c in "$LD_CKPT" "$AE_CKPT"; do
  if [ ! -s "$c" ]; then
    echo "[FATAL] missing checkpoint: $c (cwd=$(pwd)). Place it there or edit inference_ucond_notri_long.yaml." >&2
    exit 1
  fi
done
echo "[ok] checkpoints present: $LD_CKPT , $AE_CKPT"

# --- grid ---
# All grid dims are env-overridable via --export=ALL,VAR=... (or `VAR=... bash <script>`).
# Fast decisive subset (~1h on 1 A100): PROXIES="geometric rama" WS=16 SEEDS="42 43 44 45 46 47 48 49" LENGTHS=300
PROXIES="${PROXIES:-geometric rama}"
WS="${WS:-4 8 16 32}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57}"
LENGTHS="${LENGTHS:-300 400}"
RUN_BASELINE="${RUN_BASELINE:-1}"   # set 0 to skip the no-throttle baseline cells
NSTEPS=400
ROOT=results/geom_lookahead_sweep
PRIORS=steering/throttle_priors/ca_pseudo_rama.pt
CFGDIR=steering/config/geometric_lookahead/perres_frontier
mkdir -p "$CFGDIR"

# Per-proxy beta, calibrated to s=0.2 at the per-residue ΔP p90 on a real trajectory
# (model-derived, identical model -> transfers). Geom on bond/clash load, rama on pseudo-torsion.
beta_for () { case "$1" in geometric) echo 171.4809 ;; rama) echo 0.8721 ;; *) echo 1.0 ;; esac; }

write_cfg () {  # $1=proxy $2=w $3=beta $4=per_residue(true|false) $5=cfgname -> path on stdout
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

# gen (resume-safe) + scRMSD for one cell. $1=cellname $2=cfgpath
gen_eval () {
  local cell=$1 cfg=$2
  echo "[$(date -u +%FT%TZ)] gen $cell  (cfg=$cfg)"
  "$PY" -m steering.generate \
      --proteina_config inference_ucond_notri_long --steering_config "$cfg" \
      --lengths $LENGTHS --seeds $SEEDS --nsteps $NSTEPS --skip_unguided --resume \
      --output_dir "$ROOT/$cell" --device cuda:0
  echo "[$(date -u +%FT%TZ)] scRMSD designability $cell"
  OUT_BASE=$ROOT "$PY" script_utils/run_scrmsd_steering.py \
      --cfgs "$cell" --seeds $SEEDS --lengths $LENGTHS
}

run_cell () {  # $1=proxy $2=w $3=beta  (per-residue throttle cell)
  local proxy=$1 w=$2 beta=$3
  local cfg; cfg=$(write_cfg "$proxy" "$w" "$beta" true "${proxy}res_w${w}")
  gen_eval "contact_order_${proxy}res_w${w}" "$cfg"
}

# --- no-throttle baseline reference (so the frontier has its lower curve) ---
# beta=0 -> s=exp(0)=1 -> throttle never fires == plain CO steering (the no-throttle point).
# proxy=geometric needs no prior file. Same steering.generate pipeline as the throttle cells,
# so it's fully HPC-portable (no dependency on a pre-existing unguided dir).
if [ "$RUN_BASELINE" = "1" ]; then
  echo "[$(date -u +%FT%TZ)] === baseline (no-throttle, beta=0) contact_order w=$WS ==="
  for w in $WS; do
    bcfg=$(write_cfg geometric "$w" 0.0 false "baseline_w${w}")
    gen_eval "contact_order_baseline_w${w}" "$bcfg"
  done
fi

# --- throttle cells ---
for proxy in $PROXIES; do
  beta=$(beta_for "$proxy")
  for w in $WS; do run_cell "$proxy" "$w" "$beta"; done
done

echo "[$(date -u +%FT%TZ)] === PERRES_FRONTIER_DONE ==="
echo "Verdict: python script_utils/failfast_verdict.py all   (designability vs delivered CO)"
