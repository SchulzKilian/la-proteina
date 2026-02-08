#!/bin/bash

# Stop execution immediately if any command fails
set -e 

# ==============================================================================
# 0. ROBUST PATH SETUP (Run from anywhere)
# ==============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Navigate to Project Root
cd "$PROJECT_DIR"

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

# export SLURM_NTASKS=4


# ==============================================================================
# 1. Configuration & Secrets
# ==============================================================================
# export WANDB_API_KEY="your_api_key_here"

DATA_PATH="$PROJECT_DIR/data"
CHECKPOINT_DIR="/rds/user/ks2218/hpc-work/checkpoints_laproteina"
# CHECKPOINT_DIR="$PROJECT_DIR/checkpoints_laproteina"
ENV_NAME="laproteina_env"
REQUIRED_AE_CKPT="AE1_ucond_512.ckpt"

# ==============================================================================
# 2. W&B & Environment Checks
# ==============================================================================
echo "[+] Checking Environment..."

if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  WARNING: WANDB_API_KEY is not set. Run might log as 'offline'."
    echo "    Continuing in 3 seconds..."
    sleep 3
fi

# ==============================================================================
# 3. Setup Conda Environment
# ==============================================================================
if [[ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]]; then
    echo "[+] Environment '$ENV_NAME' is already active. Skipping internal setup."
else
    CONDA_CMD="conda"
    if command -v mamba &> /dev/null; then CONDA_CMD="mamba"; fi
    source $(conda info --base)/etc/profile.d/conda.sh

    if $CONDA_CMD env list | grep -q "$ENV_NAME"; then
        echo "[+] Activating environment '$ENV_NAME'..."
        conda activate "$ENV_NAME"
    else
        echo "[+] Creating environment '$ENV_NAME'..."
        $CONDA_CMD env create -f environment.yaml
        conda activate "$ENV_NAME"
        
        echo "[+] Installing pip dependencies..."
        pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
        pip install graphein==1.7.7 --no-deps
        pip install torch_geometric torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
    fi
fi

# ==============================================================================
# 4. Data & Directory Setup
# ==============================================================================

echo "[+] Setting up Data Paths..."

# --- DEBUGGING LINES ---
echo "DEBUG INFO:"
echo "  PROJECT_DIR: '$PROJECT_DIR'"
echo "  DATA_PATH:   '$DATA_PATH'"
echo "  CHECKPOINT_DIR: '$CHECKPOINT_DIR'"
# -----------------------

mkdir -p "$DATA_PATH"
mkdir -p "$CHECKPOINT_DIR"

# Write the correct path to .env so Hydra picks it up
echo "DATA_PATH=$DATA_PATH" >> .env

# --- ProteinMPNN Weights ---
mkdir -p ProteinMPNN
cd ProteinMPNN
if [ -d "ca_model_weights" ] && [ "$(ls -A ca_model_weights)" ] && \
   [ -d "vanilla_model_weights" ] && [ "$(ls -A vanilla_model_weights)" ]; then
    echo "[+] ProteinMPNN weights found."
    cd ..
else
    echo "[+] Downloading ProteinMPNN weights..."
    rm -rf ca_model_weights vanilla_model_weights
    mkdir -p ca_model_weights vanilla_model_weights

    cd ca_model_weights
    wget -qnc https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/v_48_002.pt
    wget -qnc https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/v_48_010.pt
    wget -qnc https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/v_48_020.pt
    cd ../vanilla_model_weights
    wget -qnc https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_002.pt
    wget -qnc https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_010.pt
    wget -qnc https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_020.pt
    wget -qnc https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_030.pt
    cd ../..
fi

# ==============================================================================
# 5. Checkpoint Guard (Autoencoder)
# ==============================================================================
AE_URL="https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/ae1_ucond_512.ckpt/1.0/files?redirect=true&path=AE1_ucond_512.ckpt"

if [ ! -f "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT" ]; then
    echo "⚠️  Autoencoder missing. Attempting download from NVIDIA NGC..."
    
    wget --content-disposition "$AE_URL" \
         --output-document "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT" \
         --progress=bar:force:noscroll || echo "Wget failed."

    FILE_SIZE=$(stat -c%s "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT" 2>/dev/null || echo 0)
    
    if [ "$FILE_SIZE" -lt 1000000 ]; then
        echo "❌ CRITICAL ERROR: Download failed or file corrupted."
        rm -f "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT"
        echo "   Please download '$REQUIRED_AE_CKPT' manually from the README."
        exit 1
    else
        echo "✅ Download successful."
    fi
else
    echo "[+] Autoencoder checkpoint found."
fi

# ==============================================================================
# 6. Execution: Train
# ==============================================================================
echo "[+] Starting TRAINING..."

# Note: We are already in PROJECT_DIR because of the 'cd' at step 0.
python proteinfoundation/train.py \
    dataset=pdb/pdb_train_ucond \
    nn=local_latents_score_nn_160M \
    hydra.run.dir="logs/training/$(date +%Y%m%d_%H%M%S)" \

echo "[+] Process Complete."