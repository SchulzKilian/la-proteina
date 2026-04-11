#!/bin/bash
# Download LD3 + AE2 checkpoints to the cluster.
# Run this on the LOGIN NODE (no GPU needed, just network access).
#
# Usage:
#   bash script_utils/download_ckpts_cluster.sh

set -e

CKPT_DIR="/rds/user/ks2218/hpc-work/checkpoints_laproteina"
LD3_FNAME="LD3_ucond_notri_800.ckpt"
AE2_FNAME="AE2_ucond_800.ckpt"
LD3_PATH="$CKPT_DIR/$LD3_FNAME"
AE2_PATH="$CKPT_DIR/$AE2_FNAME"

mkdir -p "$CKPT_DIR"

# Download LD3
if [ -f "$LD3_PATH" ]; then
    echo "$LD3_FNAME already present, skipping download."
else
    echo "Downloading $LD3_FNAME..."
    wget --content-disposition \
        'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/ld3_ucond_notri_800.ckpt/1.0/files?redirect=true&path=LD3_ucond_notri_800.ckpt' \
        --output-document "$LD3_PATH"
    echo "Downloaded to $LD3_PATH"
fi

# Download AE2 (LD3 uses the 800-residue autoencoder, not AE1)
if [ -f "$AE2_PATH" ]; then
    echo "$AE2_FNAME already present, skipping download."
else
    echo "Downloading $AE2_FNAME..."
    wget --content-disposition \
        'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/ae2_ucond_800.ckpt/1.0/files?redirect=true&path=AE2_ucond_800.ckpt' \
        --output-document "$AE2_PATH"
    echo "Downloaded to $AE2_PATH"
fi

# Also make sure ProteinMPNN weights are present
if [ ! -f "ProteinMPNN/vanilla_model_weights/v_48_020.pt" ]; then
    echo "ProteinMPNN weights missing -- downloading..."
    bash script_utils/download_pmpnn_weights.sh
else
    echo "ProteinMPNN weights already present."
fi

echo ""
echo "Done. Checkpoints at:"
ls -lh "$LD3_PATH" "$AE2_PATH"
