#!/bin/bash
#SBATCH -J train_test
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=500GB               
#SBATCH --time=5:00:00


# 1. Load your personal shell config (Reliable Conda setup)
source $HOME/.bashrc

# 2. Activate the environment
conda activate laproteina_env


LOCAL_DATA="/tmp/$USER/la-proteina/data"
mkdir -p $LOCAL_DATA

echo "[+] Copying latents to local SSD: $LOCAL_DATA"
# Use rsync to copy only the processed latents. 
# This usually takes ~10-15 minutes for 300GB+ but saves hours of training time.
rsync -ah --progress /home/ks2218/la-proteina/data/processed_latents $LOCAL_DATA/

# Redirect the DATA_PATH environment variable to the node's local copy
export DATA_PATH=$LOCAL_DATA

# 3. Verify Environment
echo "Running on node: $(hostname)"
echo "Using Python: $(which python)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

# 4. Run the training test script
# We run it directly since the environment is already active
bash script_utils/full_training_test.sh "$@"