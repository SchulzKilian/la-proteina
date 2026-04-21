#!/bin/bash
#SBATCH -J train_test
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --signal=SIGUSR1@300


# 1. Load your personal shell config (Reliable Conda setup)
source $HOME/.bashrc

# 2. Activate the environment
# Activate /home env via PATH prepend (NOT `conda activate`).
# /rds-based env hangs Python startup when any Lustre OST is evicted/disconn.
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

# Define exactly where your "Gold Standard" data lives
DATA_PATH="/home/ks2218/la-proteina/data"



# 3. Verify Environment
echo "Running on node: $(hostname)"
echo "Using Python: $(which python)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn) 2>/dev/null || true

# 4. Run the training test script
# We run it directly since the environment is already active
bash script_utils/full_training_test.sh "$@"