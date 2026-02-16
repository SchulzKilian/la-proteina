#!/bin/bash
#SBATCH -J gen_eval_test
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4             
#SBATCH --time=2:00:00

# Note: I reduced gpus to 1 as generation/evaluation often doesn't require 
# multi-gpu parallelism unless your code specifically supports it. 
# If you need 4, change --gres=gpu:1 to --gres=gpu:4

# 1. Load your personal shell config
source $HOME/.bashrc

# 2. Activate the environment
conda activate laproteina_env

# 3. Configuration
# Default to the most recent directory in 'store/' if not specified
STORE_DIR="./store"
LATEST_RUN=$(ls -td "$STORE_DIR"/*/ | head -1)

# You can hardcode specific configs here if they differ from the defaults
CKPT_PATH="${LATEST_RUN}checkpoints"
CKPT_NAME="last.ckpt" # Or use logic to find last-vX.ckpt if needed
AUTOENCODER_PATH="./checkpoints_laproteina/AE1_ucond_512.ckpt"
CONFIG_NAME="inference_ucond_notri"

# 4. Verify Environment and Checkpoint
echo "----------------------------------------------------------------"
echo "Running on node: $(hostname)"
echo "Using Python: $(which python)"
echo "Run Directory: $LATEST_RUN"
echo "Checkpoint Path: $CKPT_PATH/$CKPT_NAME"
echo "Config Name: $CONFIG_NAME"
echo "----------------------------------------------------------------"

if [ ! -f "$CKPT_PATH/$CKPT_NAME" ]; then
    echo "Error: Checkpoint file not found at $CKPT_PATH/$CKPT_NAME"
    exit 1
fi

# 5. Clean previous inference runs (Optional, based on your previous script)
if [ -d "inference" ]; then
    echo "Removing previous 'inference' directory..."
    rm -r inference
fi

# 6. Run Generation
echo "Starting Generation..."
python proteinfoundation/generate.py --config_name "$CONFIG_NAME" \

# 7. Run Evaluation
echo "Starting Evaluation..."
python proteinfoundation/evaluate.py --config_name "$CONFIG_NAME"

echo "Job Complete."