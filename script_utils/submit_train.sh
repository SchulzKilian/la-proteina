#!/bin/bash
#SBATCH -J train_test
#SBATCH -A COMPUTERLAB-SL3-GPU          # <--- CHANGE to your GPU account (e.g., COMPUTERLAB-SL3-GPU)
#SBATCH -p ampere                # <--- GPU partition
#SBATCH --gres=gpu:4             # Request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24       # 16 CPUs for data loading
#SBATCH --time=12:00:00          # 4 hours (adjust as needed)
#SBATCH --output=slurm_train_%j.out

# 1. Load your personal shell config (Reliable Conda setup)
source $HOME/.bashrc

# 2. Activate the environment
conda activate laproteina_env

# 3. Verify Environment
echo "Running on node: $(hostname)"
echo "Using Python: $(which python)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

# 4. Run the training test script
# We run it directly since the environment is already active
bash script_utils/full_training_test.sh