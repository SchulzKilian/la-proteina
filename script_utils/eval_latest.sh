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
PROJECT_NAME="test_release_diffusion" # Change this to your actual project name if different
CONFIG_NAME="inference_ucond_notri"

# Target the specific project directory
PROJECT_DIR="${STORE_ROOT}/${PROJECT_NAME}"

# Find the latest job folder (timestamp) inside the project directory
# ls -t sorts by modification time (newest first), head -1 takes the top one
LATEST_JOB_ID=$(ls -t "$PROJECT_DIR" | head -1)

# Construct the full path to the checkpoints
CKPT_DIR="${PROJECT_DIR}/${LATEST_JOB_ID}/checkpoints"
CKPT_NAME="last.ckpt" 

# Full path for the python script
FULL_CKPT_PATH="${CKPT_DIR}/${CKPT_NAME}"

# 4. Verify Environment and Checkpoint
echo "----------------------------------------------------------------"
echo "Running on node: $(hostname)"
echo "Using Python: $(which python)"
echo "Project Directory: $PROJECT_DIR"
echo "Latest Job ID: $LATEST_JOB_ID"
echo "Checkpoint File: $FULL_CKPT_PATH"
echo "Config Name: $CONFIG_NAME"
echo "----------------------------------------------------------------"

if [ ! -f "$FULL_CKPT_PATH" ]; then
    echo "Error: Checkpoint file not found at $FULL_CKPT_PATH"
    exit 1
fi

# 5. Clean previous inference runs
if [ -d "inference" ]; then
    echo "Removing previous 'inference' directory..."
    rm -r inference
fi

# 6. Run Generation
# Note: Ensure your generate.py accepts the checkpoint path argument correctly (e.g., --ckpt_path)
echo "Starting Generation..."
python proteinfoundation/generate.py \
    --config_name "$CONFIG_NAME" \
    # Add your checkpoint argument here if needed, e.g.:
    # --ckpt_path "$FULL_CKPT_PATH"

# 7. Run Evaluation
echo "Starting Evaluation..."
python proteinfoundation/evaluate.py --config_name "$CONFIG_NAME"

echo "Job Complete."