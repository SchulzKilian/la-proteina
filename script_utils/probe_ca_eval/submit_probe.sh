#!/bin/bash
#SBATCH -J ca_eval_probe
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_ca_eval_probe_%j.out

# Diagnostic probe: run the existing CA-only designability eval pipeline on
# real native PDBs to validate it. See run_ca_eval_probe.py for details.
# Expected wall-clock ~15 min on a healthy A100.

set -uo pipefail

source $HOME/.bashrc

# Activate /home env via PATH prepend (NOT `conda activate`).
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

# Make ProteinMPNN's subprocess use the same Python.
export PYTHON_EXEC="$LAPROTEINA_ENV/bin/python"

cd /home/ks2218/la-proteina

echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi | head -20

ulimit -n 65536 2>/dev/null || true

python script_utils/probe_ca_eval/run_ca_eval_probe.py
