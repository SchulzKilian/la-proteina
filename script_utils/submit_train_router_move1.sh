#!/bin/bash
#SBATCH -J router_train
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=1:00:00
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_router_train_%j.out

# Move-1 router training (E065). Single-GPU; consumes precomputed teacher
# .pt files under /rds/user/ks2218/hpc-work/store/router_teacher_data/.

source $HOME/.bashrc
set -uo pipefail

export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export PYTHON_EXEC=$LAPROTEINA_ENV/bin/python
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

ulimit -n 65536 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] Running on $(hostname); GPUs: ${CUDA_VISIBLE_DEVICES:-?}"
cd /home/ks2218/la-proteina

$PYTHON_EXEC script_utils/train_router_move1.py "$@"
RC=$?
echo "[$(date)] train exit=$RC"
exit $RC
