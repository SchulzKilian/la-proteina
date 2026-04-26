#!/bin/bash
#SBATCH -J manifold_pert
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_manifold_pert_%j.out

# Sidechain-manifold comparison: coord vs latent perturbations.
# Stage 1: encode → noise → decode → write PDBs (analysis_manifold/perturbation_experiment.py).
# Stage 2: for each (k, space) cell, run evaluate.py to fold each perturbed
#          structure with ESMFold and compute all-atom codesignability RMSD.
#
# Cells: NOISE_SCALES = (0.1, 0.3, 0.5, 1.0, 2.0); spaces = (coord, latent)
#        => 5 * 2 = 10 cells, evaluate.py --job_id in 0..9.
#
# NOTE: do NOT use `set -e` on this cluster (TaskProlog mkdir failure kills the
#       script before it executes). `set -uo pipefail` is the safe choice.
set -uo pipefail

source $HOME/.bashrc
# Activate via conda so PYTHONPATH/site-packages resolve correctly on this box.
# The HPC version of this script just prepends $LAPROTEINA_ENV/bin to PATH;
# that fails locally because the env lives at .conda/envs/, not conda_envs/.
# Temporarily disable nounset: conda's MKL activate script references
# $MKL_INTERFACE_LAYER unset, which would kill us under `set -u`.
set +u
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate laproteina_env
set -u

cd /home/ks2218/la-proteina

DATA_PATH="/home/ks2218/la-proteina/data"
# Local box has no raw processed/ dir; processed_latents/ files are
# OpenFold-ordered, nm, CA-centered and are accepted by load_protein's
# fallback branch. On HPC, switch back to /rds/user/ks2218/hpc-work/processed.
PROCESSED_DIR="/home/ks2218/la-proteina/data/pdb_train/processed_latents"
AE_CKPT="/home/ks2218/la-proteina/checkpoints_laproteina/AE1_ucond_512.ckpt"
EVAL_CONFIG="eval_manifold_perturbation"
OUT_ROOT="./inference/${EVAL_CONFIG}"

echo "Node: $(hostname)  Python: $(which python)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
ulimit -n 65536 2>/dev/null || true

# -------- Stage 1: build perturbed PDBs --------
echo "=== Stage 1: perturbation_experiment.py ==="
python analysis_manifold/perturbation_experiment.py \
    --processed-dir "$PROCESSED_DIR" \
    --ae-ckpt "$AE_CKPT" \
    --length-min 50 --length-max 300 \
    --n-per-bin 7 --n-bins 3 \
    --seed 0 \
    --out-root "$OUT_ROOT"

stage1_rc=$?
if [ $stage1_rc -ne 0 ]; then
    echo "Stage 1 failed (exit $stage1_rc); aborting."
    exit $stage1_rc
fi

# -------- Stage 2: evaluate each cell --------
# job_id 0..9 covers NOISE_SCALES * SPACES. evaluate.py asserts that
# `<pdb_path_without_extension>/` does not exist before creating it. If a prior
# run left tmp dirs behind (partial failure / OOM / timeout) the assertion
# would kill us at the first PDB. Pre-clean any leftover tmp dirs for each
# job_id before invoking evaluate.py.
echo "=== Stage 2: evaluate.py per cell ==="
for JOB_ID in 0 1 2 3 4 5 6 7 8 9; do
    echo "--- job_id=$JOB_ID ---"
    # Clean stale per-PDB tmp dirs (created by prior evaluate.py runs).
    find "$OUT_ROOT" -mindepth 2 -maxdepth 2 -type d -name "job_${JOB_ID}_*" -exec rm -rf {} + 2>/dev/null
    python proteinfoundation/evaluate.py \
        --config_name "$EVAL_CONFIG" \
        --job_id "$JOB_ID" \
        --data_path "$DATA_PATH"
    rc=$?
    echo "    exit=$rc"
done

# -------- Aggregate + plot (CPU; will run on the GPU node, but cheap) --------
echo "=== Aggregate + plot ==="
python analysis_manifold/aggregate_and_plot.py \
    --inference-root "$OUT_ROOT" \
    --eval-config "$EVAL_CONFIG"

echo "Done."
