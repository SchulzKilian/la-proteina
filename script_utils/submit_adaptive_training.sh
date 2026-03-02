#!/bin/bash
#SBATCH --job-name=laproteina_adaptive
#SBATCH --nodes=1
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=5:00:00
#SBATCH --partition=ampere    # or the partition your 'test' job uses

# Ensure we are in the project root
cd "$(dirname "$0")"

# Run the adaptive training script with your specific overrides
# I added 'dataset.datamodule.batch_size=8' as a suggestion to start
bash script_utils/full_training_adaptive.sh \
    dataset.datamodule.dataselector.min_length=100 \
    dataset.datamodule.dataselector.max_length=200 \
    dataset.datamodule.max_size=200 \
    dataset.datamodule.batch_size=16