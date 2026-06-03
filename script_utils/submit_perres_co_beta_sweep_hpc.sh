#!/bin/bash
#SBATCH -J perres_co_beta
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=02:00:00   # ~1.5h real (6 cells x ~13min: model reload + gen@nsteps400 + 8-seq MPNN + ESMFold)
#SBATCH --requeue
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_perres_co_beta_%j.out
#
# Per-residue GEOMETRIC look-ahead throttle BETA sweep on contact_order (minimize).
# Companion to submit_perres_co_frontier_hpc.sh. That run only had the per-residue throttle
# at the calibrated beta (geom 171.48), which OVER-DAMPS: delivered CO stayed ~0.09-0.10
# (~unguided) across w4->w16, sitting in the EASY band where the no-throttle baseline is
# already 10-12/12 designable. The whole-protein throttle (E098) instead reaches the HARD
# band (delivered CO 0.07-0.086, Δσ -2.0 to -2.4) where the no-throttle baseline is 0/12 and
# the throttle still keeps 4-7/12 alive — that's the only regime where a throttle comparison
# is meaningful, and the per-residue throttle currently has NO point there.
#
# This sweep LOWERS beta (less damping -> more guidance through -> lower delivered CO) to push
# the per-residue geometric throttle into E098's steered band, so per-residue vs whole-protein
# can finally be compared at matched delivered CO. See experiments.md E119 (+ its E098 cross-ref).
#
# Grid: proxy=geometric x BETA{20,40,80} x w{16,32} x seeds{42..49} x L{300}, nsteps=400.
#   (default beta 171.48 w16 already on disk as contact_order_geometricres_w16 from E119;
#    it is the high-beta anchor of this same sweep.)
# Cell name ENCODES beta -> contact_order_geomres_w${w}_b${beta} (no collision across betas).
# NO baseline cells (the matched-CO lower curve is already on disk: E098 contact_order_baseline_w*
# + results/geom_lookahead_sweep/analysis_property_delta.csv). Resume-safe; --requeue ok.
#
# Submit (default geom beta{20,40,80} x w{16,32} x L300 x seeds42-49 = 6 cells):
#   sbatch script_utils/submit_perres_co_beta_sweep_hpc.sh
# Add L=400 to match E098's both-lengths prop curve, or widen seeds/betas:
#   sbatch --export=ALL,LENGTHS="300 400" script_utils/submit_perres_co_beta_sweep_hpc.sh
#   sbatch --export=ALL,BETAS="10 20 40 80" script_utils/submit_perres_co_beta_sweep_hpc.sh
# If even beta=20 still over-damps (delivered CO > 0.086), drop to BETAS="5 10 20".
set -o pipefail   # NOT set -e, NOT -u (sourcing .bashrc references unset vars). See frontier script.

cd /home/ks2218/la-proteina

# --- env (PATH prepend, NOT conda activate; canonical env on /home, never /rds) ---
source $HOME/.bashrc
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
export IUPRED3_DIR=${IUPRED3_DIR:-/home/ks2218/iupred3}
export MPNN_INPROCESS=1     # in-process ProteinMPNN (one torch import, reused)
PY=python
ulimit -n 65536 2>/dev/null || true
echo "node=$(hostname) python=$(which python) gpu=${CUDA_VISIBLE_DEVICES:-unset}"

# --- pre-flight: long-generation checkpoints (paths from inference_ucond_notri_long.yaml) ---
LD_CKPT=./checkpoints_laproteina/LD3_ucond_notri_800.ckpt   # latent flow (~2.8G)
AE_CKPT=./checkpoints_laproteina/AE2_ucond_800.ckpt         # autoencoder (~3.9G)
for c in "$LD_CKPT" "$AE_CKPT"; do
  if [ ! -s "$c" ]; then
    echo "[FATAL] missing checkpoint: $c (cwd=$(pwd))." >&2
    exit 1
  fi
done
echo "[ok] checkpoints present: $LD_CKPT , $AE_CKPT"

# --- grid (all env-overridable via --export=ALL,VAR=...) ---
PROXY="${PROXY:-geometric}"          # this sweep is geometric-only (see header)
BETAS="${BETAS:-20 40 80}"           # < calibrated 171.48 = LESS damping = more steering
WS="${WS:-16 32}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49}"   # matches the E119 per-residue cells (L=300)
LENGTHS="${LENGTHS:-300}"
NSTEPS=400
ROOT=results/geom_lookahead_sweep
PRIORS=steering/throttle_priors/ca_pseudo_rama.pt
CFGDIR=steering/config/geometric_lookahead/perres_frontier
mkdir -p "$CFGDIR"

# beta with no decimal point in the cell/cfg name (20 -> b20). Keeps dirs clean.
write_cfg () {  # $1=proxy $2=w $3=beta $4=cfgname -> path on stdout
  local proxy=$1 w=$2 beta=$3 name=$4
  local f="$CFGDIR/${name}.yaml"
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

# --- throttle beta-sweep cells (geometric, per-residue) ---
for w in $WS; do
  for beta in $BETAS; do
    cfg=$(write_cfg "$PROXY" "$w" "$beta" "geomres_w${w}_b${beta}")
    gen_eval "contact_order_geomres_w${w}_b${beta}" "$cfg"
  done
done

echo "[$(date -u +%FT%TZ)] === PERRES_CO_BETA_SWEEP_DONE ==="
echo "Verdict: python script_utils/failfast_verdict.py all   (designability vs delivered CO;"
echo "         new cells: contact_order_geomres_w{16,32}_b{20,40,80})"
