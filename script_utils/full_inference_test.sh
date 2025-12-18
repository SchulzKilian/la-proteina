#!/bin/bash

# 1. Stop the script if any command fails (Good practice)
set -e

# --- SETUP VARIABLES ---
SCRATCH_DIR="/homes/ks2218/la-proteina"
SCRATCH_DIR="$HOME/Programming/la-proteina"
ENV_PATH="$SCRATCH_DIR/my_env"
CHECKPOINT_DIR="$SCRATCH_DIR/checkpoints_laproteina"

echo "Starting setup..."

# 2. Download ProteinMPNN weights
echo "Downloading ProteinMPNN weights..."
bash script_utils/download_pmpnn_weights.sh

# 3. Handle Conda Environment

source $(conda info --base)/etc/profile.d/conda.sh

if conda info --envs | grep -q "^laproteina_env "; then
    echo "Environment exists. Updating..."github.com/SchulzKilian/la-proteinagithub.com/SchulzKilian/la-proteina
    conda env update --name "laproteina_env" --file environment.yaml --prune
else
    echo "Creating new environment..."
    conda env create --name "laproteina_env" --file environment.yaml


fi

conda activate laproteina_env

# 4. DOWNLOAD CHECKPOINTS TO SCRATCH 
echo "Setting up checkpoints in scratch space..."

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "✅ Checkpoint folder exists at $CHECKPOINT_DIR. Skipping download."
else
    echo "⬇️  Checkpoint folder not found. Creating and downloading..."
    
    # Create the directory
    mkdir -p "$CHECKPOINT_DIR"

    # Download files
    curl -L 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/ld1_ucond_notri_512.ckpt/1.0/files?redirect=true&path=LD1_ucond_notri_512.ckpt' -o "$CHECKPOINT_DIR/LD1_ucond_notri_512.ckpt"
    curl -L 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/ae1_ucond_512.ckpt/1.0/files?redirect=true&path=AE1_ucond_512.ckpt' -o "$CHECKPOINT_DIR/AE1_ucond_512.ckpt"
fi
ln -sfn "$CHECKPOINT_DIR" ./checkpoints_laproteina

# 5. Run the evaluation script
echo "Running generation and evaluation..."
bash script_utils/gen_n_eval.sh

echo "Job Done!"
