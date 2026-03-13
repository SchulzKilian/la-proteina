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


REMOTE_DATA="/home/ks2218/la-proteina/data/pdb_train"
LOCAL_DATA="/tmp/$USER/la-proteina/data/pdb_train"

mkdir -p "$(dirname "$LOCAL_DATA")"

echo "[+] Syncing ENTIRE data folder to local SSD: $LOCAL_DATA"
# Syncing the entire folder ensures CSVs, FASTA, and clustering files are present
rsync -ah --progress "$REMOTE_DATA/" "$LOCAL_DATA/"

# 2. Export the DATA_PATH to the local SSD copy
export DATA_PATH="$LOCAL_DATA"
# Redirect the DATA_PATH environment variable to the node's local copy


# 3. Verify Environment
echo "Running on node: $(hostname)"
echo "Using Python: $(which python)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

# 4. Run the training test script
# We run it directly since the environment is already active
bash script_utils/full_training_test.sh "$@"