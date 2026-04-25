#!/bin/bash
#SBATCH -J dense_t_eval
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=slurm_dense_t_eval_%j.out

# Dense-t fixed-seed val comparison: step-2204 raw vs step-2457 raw checkpoints.
# Diagnostic for the late-training val-loss uptick — is it signal or noise?

set -uo pipefail

export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

# Load .env-style vars (DATA_PATH for the dataset). Reuse the train script's idea.
if [ -f /home/ks2218/la-proteina/.env ]; then
    set -o allexport; source /home/ks2218/la-proteina/.env; set +o allexport
fi
export DATA_PATH=${DATA_PATH:-/rds/user/ks2218/hpc-work}

cd /home/ks2218/la-proteina

echo "=== Dense-t fixed-seed val eval ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date), DATA_PATH=$DATA_PATH"

python /home/ks2218/la-proteina/analysis_cheap_diagnostics/dense_t_eval.py \
    --n_batches 32 \
    --n_t 20 \
    --t_min 0.05 \
    --t_max 0.95 \
    --seed_base 42

rc=$?
echo "Eval finished (rc=$rc) at $(date)"
exit $rc
