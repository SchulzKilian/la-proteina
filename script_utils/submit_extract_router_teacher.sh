#!/bin/bash
#SBATCH -J router_teach
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=0:30:00
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_router_teach_%j.out

# Extract dense-attention top-64 + per-layer hidden-state teacher data for
# Move-1 routing-probe (E065). Single-GPU forward-only.
#
# Usage:
#   sbatch script_utils/submit_extract_router_teacher.sh            # full 500
#   sbatch script_utils/submit_extract_router_teacher.sh --smoke    # 5 proteins
# Extra args after the script name pass through to extract_router_teacher_data.py.
#
# Conventions (per CLAUDE.md):
#   - PATH-prepend the env (NEVER `conda activate` — Lustre hang risk).
#   - set -uo pipefail (NOT set -e — TaskProlog mkdir bug kills set -e scripts).
#   - export PYTHON_EXEC (downstream tooling needs it).
#   - --exclude=gpu-q-43 (broken GPU; afterany re-routes there otherwise).

source $HOME/.bashrc
set -uo pipefail

export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export PYTHON_EXEC=$LAPROTEINA_ENV/bin/python
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

echo "[$(date)] Running on $(hostname); GPUs: ${CUDA_VISIBLE_DEVICES:-?}"
echo "[$(date)] Python: $(which python)"

ulimit -n 65536 2>/dev/null || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/ks2218/la-proteina

$PYTHON_EXEC script_utils/extract_router_teacher_data.py "$@"
RC=$?
echo "[$(date)] extract exit=$RC"
exit $RC
