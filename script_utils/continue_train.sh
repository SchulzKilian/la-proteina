#!/bin/bash
#SBATCH -J train_resume
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0                
#SBATCH --time=5:00:00

# 1. Load shell config and activate environment
source $HOME/.bashrc
conda activate laproteina_env

# 2. Determine Checkpoint Path
# If an argument is provided ($1), use it. Otherwise, find the latest.
if [ -n "$1" ]; then
    CHECKPOINT_PATH="$1"
    echo "[+] Using provided checkpoint: $CHECKPOINT_PATH"
else
    echo "[!] No checkpoint provided. Searching for latest..."
    
    # Logic adapted from eval_latest.sh
    STORE_ROOT="./store"
    PROJECT_NAME="test_release_diffusion" 
    PROJECT_DIR="${STORE_ROOT}/${PROJECT_NAME}"

    if [ -d "$PROJECT_DIR" ]; then
        # Sort by time and pick the newest directory
        LATEST_JOB_ID=$(ls -t "$PROJECT_DIR" | head -1)
        CHECKPOINT_PATH="${PROJECT_DIR}/${LATEST_JOB_ID}/checkpoints/last.ckpt"
        echo "[+] Found latest checkpoint: $CHECKPOINT_PATH"
    else
        echo "Error: Project directory $PROJECT_DIR not found. Cannot find latest checkpoint."
        exit 1
    fi
fi

# 3. Verify the file exists before starting
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found at $CHECKPOINT_PATH"
    exit 1
fi

# 4. Run the training resume script
bash script_utils/full_training_resume.sh "$CHECKPOINT_PATH"