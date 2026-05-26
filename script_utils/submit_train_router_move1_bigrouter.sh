#!/bin/bash
#SBATCH -J router_big
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=1:30:00
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_router_big_%j.out

# E068: Probe 2 — bigger router (hidden_dim=256, score_dim=64, +mlp_block).
# Whether to also keep --pair_features depends on Probe 1 outcome:
#   - Probe 1 success → pass --pair_features when submitting
#   - Probe 1 flat   → don't pass it; we want to isolate the capacity axis

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

# Note: B=8 at hidden_dim=256 may approach memory limit at N=500. Drop to B=4
# if needed — the OOM-guard in train_router_move1.py logs a warning and skips
# but won't crash.
$PYTHON_EXEC script_utils/train_router_move1.py \
    --hidden_dim 256 \
    --score_dim 64 \
    --mlp_block \
    --out_dir results/router_move1_bigrouter \
    "$@"
RC=$?
echo "[$(date)] train_big exit=$RC"
exit $RC
