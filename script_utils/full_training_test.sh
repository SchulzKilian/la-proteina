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

export SLURM_NTASKS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_COMPILE_DISABLE=1

# ==============================================================================
# 1. Configuration & Defaults
# ==============================================================================
# DEFAULT VALUES
DATA_PATH="$PROJECT_DIR/data"
CHECKPOINT_DIR="/rds/user/ks2218/hpc-work/checkpoints_laproteina"
ENV_NAME="laproteina_env"
REQUIRED_AE_CKPT="AE1_ucond_512.ckpt"

# PARSE COMMAND LINE OVERRIDES
# Usage: ./script.sh -d /new/data -c /new/ckpt -e new_env -- hydra_args=value
while getopts "d:c:e:" opt; do
  case $opt in
    d) DATA_PATH="$OPTARG" ;;
    c) CHECKPOINT_DIR="$OPTARG" ;;
    e) ENV_NAME="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Shift off the options consumed by getopts so that $@ only contains 
# the remaining training/hydra arguments.
shift $((OPTIND-1))

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
    
    # Ensure conda functions are available in subshell
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"

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

echo "DEBUG INFO:"
echo "  PROJECT_DIR:    '$PROJECT_DIR'"
echo "  DATA_PATH:      '$DATA_PATH'"
echo "  CHECKPOINT_DIR: '$CHECKPOINT_DIR'"
echo "  ENV_NAME:       '$ENV_NAME'"

mkdir -p "$DATA_PATH"
mkdir -p "$CHECKPOINT_DIR"

# Update/Append the path to .env for Hydra
# Note: Using sed to update if it exists, or appending if it doesn't
if grep -q "DATA_PATH=" .env 2>/dev/null; then
    sed -i "s|DATA_PATH=.*|DATA_PATH=$DATA_PATH|" .env
else
    echo "DATA_PATH=$DATA_PATH" >> .env
fi

# --- ProteinMPNN Weights ---
mkdir -p ProteinMPNN
pushd ProteinMPNN > /dev/null
if [ -d "ca_model_weights" ] && [ "$(ls -A ca_model_weights)" ] && \
   [ -d "vanilla_model_weights" ] && [ "$(ls -A vanilla_model_weights)" ]; then
    echo "[+] ProteinMPNN weights found."
else
    echo "[+] Downloading ProteinMPNN weights..."
    rm -rf ca_model_weights vanilla_model_weights
    mkdir -p ca_model_weights vanilla_model_weights

    wget -qnc -P ca_model_weights https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/v_48_002.pt
    wget -qnc -P ca_model_weights https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/v_48_010.pt
    wget -qnc -P ca_model_weights https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/v_48_020.pt
    
    wget -qnc -P vanilla_model_weights https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_002.pt
    wget -qnc -P vanilla_model_weights https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_010.pt
    wget -qnc -P vanilla_model_weights https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_020.pt
    wget -qnc -P vanilla_model_weights https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_030.pt
fi
popd > /dev/null

# ==============================================================================
# 5. Checkpoint Guard (Autoencoder)
# ==============================================================================
AE_URL="https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/ae1_ucond_512.ckpt/1.0/files?redirect=true&path=AE1_ucond_512.ckpt"

if [ ! -f "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT" ]; then
    echo "⚠️  Autoencoder missing in $CHECKPOINT_DIR. Attempting download..."
    
    wget --content-disposition "$AE_URL" \
         --output-document "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT" \
         --progress=bar:force:noscroll || echo "Wget failed."

    FILE_SIZE=$(stat -c%s "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT" 2>/dev/null || echo 0)
    
    if [ "$FILE_SIZE" -lt 1000000 ]; then
        echo "❌ CRITICAL ERROR: Download failed or file corrupted."
        rm -f "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT"
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
export SLURM_NTASKS_PER_NODE=$SLURM_NTASKS

# "$@" now contains ONLY the arguments that weren't picked up by getopts
srun python proteinfoundation/train.py \
    dataset=pdb/pdb_train_ucond \
    nn=local_latents_score_nn_160M \
    hydra.run.dir="logs/training/$(date +%Y%m%d_%H%M%S)" \
    "$@"

echo "[+] Process Complete."