#!/bin/bash
#SBATCH -J router_inflight
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=2:30:00
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_router_inflight_%j.out

# In-flight router training: dense teacher signal captured per-step from
# the canonical PDB dataloader. No offline teacher .pt files.
# Defaults: bigrouter + pair_features + use_t_emb.

source $HOME/.bashrc
set -uo pipefail

export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export PYTHON_EXEC=$LAPROTEINA_ENV/bin/python
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

ulimit -n 65536 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# DATA_PATH is needed by the canonical YAML's ${oc.env:DATA_PATH} interpolation.
: "${DATA_PATH:=/home/ks2218/la-proteina/data}"
export DATA_PATH

echo "[$(date)] $(hostname); GPUs: ${CUDA_VISIBLE_DEVICES:-?}"
echo "[$(date)] DATA_PATH=$DATA_PATH"
cd /home/ks2218/la-proteina

$PYTHON_EXEC script_utils/train_router_move1_inflight.py "$@"
RC=$?
echo "[$(date)] inflight train exit=$RC"
exit $RC
