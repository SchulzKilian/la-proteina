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

# Define exactly where your "Gold Standard" data lives
REMOTE_DATA="/home/ks2218/la-proteina/data"
LOCAL_DATA="/tmp/$USER/la-proteina/data"

mkdir -p "$LOCAL_DATA"

echo "[+] Syncing ALL metadata and latents to local SSD..."
# The trailing slash on REMOTE_DATA/ is important to copy contents, not the folder itself
rsync -avh --progress "$REMOTE_DATA/" "$LOCAL_DATA/"

# Explicitly set the path for the training script
export DATA_PATH="$LOCAL_DATA"
# Redirect the DATA_PATH environment variable to the node's local copy
echo $REMOTE_DATA
echo $LOCAL_DATA

# 3. Verify Environment
echo "Running on node: $(hostname)"
echo "Using Python: $(which python)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

# 4. Run the training test script
# We run it directly since the environment is already active
bash script_utils/full_training_test.sh "$@"