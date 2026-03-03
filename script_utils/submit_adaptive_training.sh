#!/bin/bash
#SBATCH --job-name=laproteina_adaptive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=5:00:00
#SBATCH --partition=ampere
#SBATCH --output=logs/slurm/%j.out

source $HOME/.bashrc
conda activate laproteina_env
# 4. Performance & Memory Tuning
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_COMPILE_DISABLE=1

# 5. Execution
# We use the adaptive script to ensure the correct AE checkpoint is selected
# based on the 200-residue limit.
echo "[+] Submitting Adaptive Training Job with max_length=300 (to trigger AE2 selection)"
bash script_utils/full_training_adaptive.sh \
    dataset.datamodule.dataselector.min_length=100 \
    dataset.datamodule.dataselector.max_length=300 \
    dataset.datamodule.batch_size=12 \
