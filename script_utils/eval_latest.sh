#!/bin/bash
#SBATCH -J gen_eval_test
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4             
#SBATCH --time=2:00:00

# 1. Load shell config and activate environment
source $HOME/.bashrc
conda activate laproteina_env

# 2. Configuration
STORE_ROOT="./store"
PROJECT_NAME="test_release_diffusion" 
CONFIG_NAME="inference_ucond_notri"

# 3. Determine Full Checkpoint Path
if [ -n "$1" ]; then
    # Use provided path relative to STORE_ROOT
    FULL_CKPT_PATH="${STORE_ROOT}/$1"
    echo "[+] Using provided model path: $FULL_CKPT_PATH"
else
    # Fallback to Latest logic
    echo "[!] No path provided. Searching for latest checkpoint..."
    PROJECT_DIR="${STORE_ROOT}/${PROJECT_NAME}"
    if [ ! -d "$PROJECT_DIR" ]; then
        echo "Error: Project directory $PROJECT_DIR not found."
        exit 1
    fi
    LATEST_JOB_ID=$(ls -t "$PROJECT_DIR" | head -1)
    FULL_CKPT_PATH="${PROJECT_DIR}/${LATEST_JOB_ID}/checkpoints/last-EMA.ckpt"
fi

# 4. Verify file existence
if [ ! -f "$FULL_CKPT_PATH" ]; then
    echo "Error: Checkpoint file not found at $FULL_CKPT_PATH"
    exit 1
fi

# 5. Split path for generate.py (which expects ckpt_path and ckpt_name separately)
# dirname gets the folder, basename gets the filename
CKPT_DIR_OVERRIDE=$(dirname "$FULL_CKPT_PATH")
CKPT_NAME_OVERRIDE=$(basename "$FULL_CKPT_PATH")

echo "----------------------------------------------------------------"
echo "Checkpoint Folder: $CKPT_DIR_OVERRIDE"
echo "Checkpoint Name:   $CKPT_NAME_OVERRIDE"
echo "----------------------------------------------------------------"

# 6. Clean previous inference runs
if [ -d "inference" ]; then rm -r inference; fi

# 7. Run Generation
# Pass overrides directly to Hydra (no '--' for ckpt_path and ckpt_name)
echo "Starting Generation..."
python proteinfoundation/generate.py \
    --config_name "$CONFIG_NAME" \
    ++ckpt_path="$CKPT_DIR_OVERRIDE" \
    ++ckpt_name="$CKPT_NAME_OVERRIDE"

# 8. Run Evaluation
echo "Starting Evaluation..."
python proteinfoundation/evaluate.py --config_name "$CONFIG_NAME"

echo "Job Complete."