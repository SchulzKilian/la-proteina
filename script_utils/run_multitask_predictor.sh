#!/bin/bash
#SBATCH -J multitask_t1
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH -o slurm_multitask_t1_%j.out

# Do NOT use set -e on this cluster (SLURM TaskProlog mkdir fails)
set -uo pipefail

source $HOME/.bashrc
conda activate laproteina_env

echo "Running on node: $(hostname)"
echo "Using Python: $(which python)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"
nvidia-smi

ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn) 2>/dev/null || true

cd /home/ks2218/la-proteina/laproteina_steerability

# Parse arguments
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test) EXTRA_ARGS="$EXTRA_ARGS --smoke-test"; shift ;;
        --folds) EXTRA_ARGS="$EXTRA_ARGS --folds $2"; shift 2 ;;
        --batch-size) EXTRA_ARGS="$EXTRA_ARGS --batch-size $2"; shift 2 ;;
        --max-epochs) EXTRA_ARGS="$EXTRA_ARGS --max-epochs $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "Extra args: $EXTRA_ARGS"
python -m src.multitask_predictor.run --config config/default.yaml $EXTRA_ARGS
