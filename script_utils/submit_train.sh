#!/bin/bash
#SBATCH -J train_test
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G              
#SBATCH --time=3:00:00


# 1. Load your personal shell config (Reliable Conda setup)
source $HOME/.bashrc

# 2. Activate the environment
conda activate laproteina_env

# Define exactly where your "Gold Standard" data lives
DATA_PATH="/home/ks2218/la-proteina/data"



# 3. Verify Environment
echo "Running on node: $(hostname)"
echo "Using Python: $(which python)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

# 4. Run the training test script
# We run it directly since the environment is already active
bash script_utils/full_training_test.sh "$@"