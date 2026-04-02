#!/bin/bash
set -e

CKPT_DIR="${CHECKPOINT_DIR:-./checkpoints_laproteina}"
LD3_FNAME="LD3_ucond_notri_800.ckpt"
AE2_FNAME="AE2_ucond_800.ckpt"
LD3_PATH="$CKPT_DIR/$LD3_FNAME"
AE2_PATH="$CKPT_DIR/$AE2_FNAME"

mkdir -p "$CKPT_DIR"

# Download LD3
if [ -f "$LD3_PATH" ]; then
    echo "✅ $LD3_FNAME already present, skipping download."
else
    echo "⬇️  Downloading $LD3_FNAME..."
    wget --content-disposition \
        'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/ld3_ucond_notri_800.ckpt/1.0/files?redirect=true&path=LD3_ucond_notri_800.ckpt' \
        --output-document "$LD3_PATH"
    echo "✅ Downloaded to $LD3_PATH"
fi

# Download AE2 (LD3 uses the 800-residue autoencoder, not AE1)
if [ -f "$AE2_PATH" ]; then
    echo "✅ $AE2_FNAME already present, skipping download."
else
    echo "⬇️  Downloading $AE2_FNAME..."
    wget --content-disposition \
        'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/ae2_ucond_800.ckpt/1.0/files?redirect=true&path=AE2_ucond_800.ckpt' \
        --output-document "$AE2_PATH"
    echo "✅ Downloaded to $AE2_PATH"
fi

# Measure straightness — use nres=400 (middle of LD3's 300-800 range)
echo ""
echo "📐 Measuring vector field straightness for LD3..."
python script_utils/measure_field_straightness.py \
    --ckpt_dir     "$CKPT_DIR" \
    --ckpt_name    "$LD3_FNAME" \
    --ae_ckpt_name "$AE2_FNAME" \
    --nsamples 80 \
    --nres 400 \
    --nsteps 800 \
    --out_json "$CKPT_DIR/straightness_ld3.json"

echo ""
echo "Done. Results saved to $CKPT_DIR/straightness_ld3.json"
