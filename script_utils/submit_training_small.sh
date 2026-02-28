#!/bin/bash
#SBATCH -J train_small
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=500GB               
#SBATCH --time=12:00:00 # Increased time for a real run

source $HOME/.bashrc
conda activate laproteina_env

# Run the small-protein training script
bash script_utils/full_training_small.sh "$@"