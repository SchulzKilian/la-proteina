#!/bin/bash
#SBATCH -J router_coords
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=0:30:00
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_router_coords_%j.out

# Sidecar coords extraction for Probe 1 (E067) pair-distance features.
# Same recipe as the main extraction; just dumps bb_ca per (protein, t).

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

$PYTHON_EXEC script_utils/extract_router_teacher_coords.py "$@"
RC=$?
echo "[$(date)] coords-extract exit=$RC"
exit $RC
