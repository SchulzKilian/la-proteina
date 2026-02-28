#!/bin/bash
set -e 

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# CONFIGURATION
DATA_PATH="$PROJECT_DIR/data"
CHECKPOINT_DIR="/rds/user/ks2218/hpc-work/checkpoints_laproteina"
REQUIRED_AE_CKPT="AE3_motif.ckpt"
# API URL for the AE3 checkpoint (motif scaffolding optimized)
AE_URL="https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/ae3_motif.ckpt/1.0/files?redirect=true&path=AE3_motif.ckpt"

export SLURM_NTASKS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_COMPILE_DISABLE=1

mkdir -p "$DATA_PATH"
mkdir -p "$CHECKPOINT_DIR"

# .env update for Hydra
if grep -q "DATA_PATH=" .env 2>/dev/null; then
    sed -i "s|DATA_PATH=.*|DATA_PATH=$DATA_PATH|" .env
else
    echo "DATA_PATH=$DATA_PATH" >> .env
fi

# Checkpoint Guard for AE3
if [ ! -f "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT" ]; then
    echo "⚠️  AE3 missing. Attempting download..."
    wget --no-check-certificate \
         --content-disposition "$AE_URL" \
         --output-document "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT" \
         --progress=bar:force:noscroll || exit 1
    echo "✅ AE3 Download successful."
else
    echo "[+] AE3 checkpoint found."
fi

# EXECUTION with overrides for length 25-100
echo "[+] Starting TRAINING (Size 25-100)..."
srun python proteinfoundation/train.py \
    dataset=pdb/pdb_train_ucond \
    nn=local_latents_score_nn_160M \
    autoencoder_ckpt_path="$CHECKPOINT_DIR/$REQUIRED_AE_CKPT" \
    dataset.datamodule.dataselector.min_length=25 \
    dataset.datamodule.dataselector.max_length=100 \
    hydra.run.dir="logs/training/small_$(date +%Y%m%d_%H%M%S)" \
    "$@"