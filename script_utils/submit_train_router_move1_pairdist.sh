#!/bin/bash
#SBATCH -J router_pair
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=1:00:00
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_router_pair_%j.out

# E067: Probe 1 — pair-distance RBF features on top of baseline router.
# Same recipe as E065 train; only deltas are --pair_features and --out_dir.

source $HOME/.bashrc
set -uo pipefail

export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export PYTHON_EXEC=$LAPROTEINA_ENV/bin/python
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

ulimit -n 65536 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] $(hostname); GPUs: ${CUDA_VISIBLE_DEVICES:-?}"
cd /home/ks2218/la-proteina

$PYTHON_EXEC script_utils/train_router_move1.py \
    --pair_features \
    --out_dir results/router_move1_pairdist \
    "$@"
RC=$?
echo "[$(date)] train_pair exit=$RC"
exit $RC
