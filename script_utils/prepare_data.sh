#!/bin/bash

# Stop execution immediately if any command fails
set -e 

# Load .env variables if present
if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

# ==============================================================================
# 1. Configuration & Directories
# ==============================================================================
PROJECT_DIR="$(pwd)"

# Define your paths (Adjust these if your cluster paths differ)
DATA_PATH="$PROJECT_DIR/data"
# DATA_PATH="/rds/user/ks2218/hpc-work/la-proteina/data/pdb_train/raw"

CHECKPOINT_DIR="/rds/user/ks2218/hpc-work/checkpoints_laproteina"
# CHECKPOINT_DIR="$PROJECT_DIR/checkpoints_laproteina" 

ENV_NAME="laproteina_env"
REQUIRED_AE_CKPT="AE1_ucond_512.ckpt"

# Ensure directories exist
mkdir -p "$DATA_PATH"
mkdir -p "$CHECKPOINT_DIR"

# ==============================================================================
# 2. Setup Conda Environment
# ==============================================================================
echo "[+] Setting up Environment..."
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
    pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
    pip install graphein==1.7.7 --no-deps
    pip install torch_geometric torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
fi

# ==============================================================================
# 3. Download Dependencies (Weights & Checkpoints)
# ==============================================================================

# --- ProteinMPNN Weights ---
echo "[+] Checking ProteinMPNN weights..."
mkdir -p ProteinMPNN
cd ProteinMPNN
if [ ! -d "ca_model_weights" ] || [ ! -d "vanilla_model_weights" ]; then
    echo "Downloading ProteinMPNN weights..."
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
else
    cd .. # Back to script root
fi

# --- Autoencoder Checkpoint ---
echo "[+] Checking Autoencoder checkpoint..."
AE_URL="https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/ae1_ucond_512.ckpt/1.0/files?redirect=true&path=AE1_ucond_512.ckpt"
if [ ! -f "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT" ]; then
    echo "Downloading AE checkpoint..."
    wget --content-disposition "$AE_URL" \
         --output-document "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT" \
         --progress=bar:force:noscroll || echo "Wget failed."
fi

# ==============================================================================
# 4. Execution: Run Data Prep (CPU Mode)
# ==============================================================================
echo "[+] Starting DATA PREPARATION script..."

# We execute the python script you already created.
# We override:
#   1. data_dir -> to point to your $DATA_PATH
#   2. num_workers -> 16 (for parallel CPU processing)
#   3. accelerator -> cpu (to avoid NVME errors)
#   4. nolog -> true (to disable WandB completely)

python prepare_data.py \
    dataset=pdb/pdb_train_ucond \
    dataset.datamodule.data_dir="$DATA_PATH" \
    dataset.datamodule.num_workers=16 \
    +nolog=true \
    ++hardware.accelerator=cpu \
    ++hardware.ngpus_per_node_=1

echo "[+] Data Preparation Complete."