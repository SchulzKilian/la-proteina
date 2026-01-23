#!/bin/bash

# ==============================================================================
# Setup Variables
# ==============================================================================
# Define where the protein data will be stored
export DATA_PATH="$(pwd)/data"
mkdir -p "$DATA_PATH"

# Setup the .env file required by the codebase
echo "Creating .env file with DATA_PATH..."
echo "DATA_PATH=$DATA_PATH" > .env

# ==============================================================================
# Environment Setup (Conda + Pip)
# ==============================================================================
ENV_NAME="laproteina_env"
CONDA_CMD="conda"

# Check if mamba is available (preferred by README), otherwise use conda
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
fi

echo "Checking for conda environment: $ENV_NAME..."

if $CONDA_CMD env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' exists. Activating..."
    # Source conda base to ensure 'activate' works in script
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate "$ENV_NAME"
else
    echo "Environment '$ENV_NAME' not found. Creating from environment.yaml..."
    
    # Create environment
    $CONDA_CMD env create -f environment.yaml
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate "$ENV_NAME"
    
    # Install specific pip dependencies as per La-Proteina README
    echo "Installing specific PyTorch and Graphein dependencies..."
    
    # Note: These versions are strictly from the La-Proteina setup instructions
    pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
    pip install graphein==1.7.7 --no-deps
    pip install torch_geometric torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
fi

# ==============================================================================
# Additional Setup (Weights & Checkpoints)
# ==============================================================================
echo "Downloading ProteinMPNN weights (required for evaluation)..."
if [ -f "script_utils/download_pmpnn_weights.sh" ]; then
    bash script_utils/download_pmpnn_weights.sh
else
    echo "Warning: script_utils/download_pmpnn_weights.sh not found. Skipping."
fi

# Check for Autoencoder checkpoints (Critical for training)
# The README states these must be manually downloaded to ./checkpoints_laproteina
if [ ! -d "checkpoints_laproteina" ] || [ -z "$(ls -A checkpoints_laproteina 2>/dev/null)" ]; then
    echo "----------------------------------------------------------------"
    echo "WARNING: Model checkpoints missing in ./checkpoints_laproteina"
    echo "Training requires autoencoder checkpoints (e.g., AE1_ucond_512.ckpt)."
    echo "Please download them from the links in the README and place them"
    echo "in the 'checkpoints_laproteina' folder before running full training."
    echo "----------------------------------------------------------------"
    # Create the directory just in case
    mkdir -p checkpoints_laproteina
fi

# ==============================================================================
# Run Training
# ==============================================================================
echo "Starting training..."
echo "Using Data Path: $DATA_PATH"

# We override the dataset to 'pdb_train_ucond' to ensure data is automatically 
# downloaded and processed (AFDB requires manual ID lists).
# We also set the neural network architecture as required for unconditional training.

python proteinfoundation/train.py \
    dataset=pdb/pdb_train_ucond \
    nn=local_latents_score_nn_160M \
    hydra.run.dir=.