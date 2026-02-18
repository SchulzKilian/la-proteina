#!/bin/bash
#SBATCH -J gen_eval_test
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4             
#SBATCH --time=2:00:00

# 1. Load your personal shell config
source $HOME/.bashrc

# 2. Activate the environment
conda activate laproteina_env

# 3. Configuration
STORE_ROOT="./store"
PROJECT_NAME="test_release_diffusion" 
CONFIG_NAME="inference_ucond_notri"

# 4. Determine Checkpoint Path
if [ -n "$1" ]; then
    # If an argument is provided, treat it as the path relative to STORE_ROOT
    FULL_CKPT_PATH="${STORE_ROOT}/$1"
    echo "[+] Using provided model path: $FULL_CKPT_PATH"
else
    # Fallback to "Latest" logic
    echo "[!] No path provided. Searching for latest checkpoint in $PROJECT_NAME..."
    
    PROJECT_DIR="${STORE_ROOT}/${PROJECT_NAME}"
    
    if [ ! -d "$PROJECT_DIR" ]; then
        echo "Error: Project directory $PROJECT_DIR does not exist."
        exit 1
    fi

    # Find the latest job folder (timestamp)
    LATEST_JOB_ID=$(ls -t "$PROJECT_DIR" | head -1)
    CKPT_NAME="last.ckpt" 
    FULL_CKPT_PATH="${PROJECT_DIR}/${LATEST_JOB_ID}/checkpoints/${CKPT_NAME}"
    
    echo "[+] Latest Job ID found: $LATEST_JOB_ID"
fi

# 5. Verify Checkpoint exists
if [ ! -f "$FULL_CKPT_PATH" ]; then
    echo "Error: Checkpoint file not found at $FULL_CKPT_PATH"
    exit 1
fi

echo "----------------------------------------------------------------"
echo "Running on node: $(hostname)"
echo "Checkpoint File: $FULL_CKPT_PATH"
echo "Config Name: $CONFIG_NAME"
echo "----------------------------------------------------------------"

# 6. Clean previous inference runs
if [ -d "inference" ]; then
    echo "Removing previous 'inference' directory..."
    rm -r inference
fi

# 7. Run Generation
# Passing the detected/provided path to the script
echo "Starting Generation..."
python proteinfoundation/generate.py \
    --config_name "$CONFIG_NAME" \
    --ckpt_path "$FULL_CKPT_PATH"

# 8. Run Evaluation
echo "Starting Evaluation..."
python proteinfoundation/evaluate.py --config_name "$CONFIG_NAME"

echo "Job Complete."