#!/bin/bash
set -e 

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 1. Defaults
DATA_PATH="$PROJECT_DIR/data"
CHECKPOINT_DIR="/rds/user/ks2218/hpc-work/checkpoints_laproteina"
MAX_LEN=512 # Default fallback

# 2. Parse max_length from command line arguments to pick AE
for arg in "$@"; do
    if [[ $arg == dataset.datamodule.dataselector.max_length=* ]]; then
        MAX_LEN="${arg#*=}"
    fi
done

# 3. Selection Logic based on La-Proteina Model Card
if [ "$MAX_LEN" -le 256 ]; then
    REQUIRED_AE_CKPT="AE3_motif.ckpt"
    AE_URL="https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/ae3_motif.ckpt/1.0/files?redirect=true&path=AE3_motif.ckpt"
    echo "[+] Target Length <= 256: Selecting AE3 (Motif/Small optimized)"
elif [ "$MAX_LEN" -le 512 ]; then
    REQUIRED_AE_CKPT="AE1_ucond_512.ckpt"
    AE_URL="https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/ae1_ucond_512.ckpt/1.0/files?redirect=true&path=AE1_ucond_512.ckpt"
    echo "[+] Target Length <= 512: Selecting AE1 (Standard Unconditional)"
else
    REQUIRED_AE_CKPT="AE2_ucond_800.ckpt"
    AE_URL="https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/ae2_ucond_800.ckpt/1.0/files?redirect=true&path=AE2_ucond_800.ckpt"
    echo "[+] Target Length > 512: Selecting AE2 (Long-chain optimized)"
fi

mkdir -p "$CHECKPOINT_DIR"

# 4. Robust Checkpoint Guard
if [ ! -f "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT" ] || [ ! -s "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT" ]; then
    echo "⚠️  $REQUIRED_AE_CKPT missing or empty. Downloading..."
    rm -f "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT"
    wget --no-check-certificate --content-disposition "$AE_URL" \
         --output-document "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT" \
         --progress=bar:force:noscroll || exit 1
    
    # Verify size (must be > 1MB)
    FILE_SIZE=$(stat -c%s "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT" 2>/dev/null || echo 0)
    if [ "$FILE_SIZE" -lt 1000000 ]; then
        echo "❌ ERROR: Download failed."
        rm -f "$CHECKPOINT_DIR/$REQUIRED_AE_CKPT"
        exit 1
    fi
fi

# 5. Execution
# Pass all original arguments "$@" so your length overrides reach python
python proteinfoundation/train.py \
    autoencoder_ckpt_path="$CHECKPOINT_DIR/$REQUIRED_AE_CKPT" \
    hydra.run.dir="logs/training/adaptive_$(date +%Y%m%d_%H%M%S)" \
    "$@"