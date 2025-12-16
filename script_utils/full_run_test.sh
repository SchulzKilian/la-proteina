#!/bin/bash

# 1. Stop the script if any command fails (Good practice)
set -e

# --- SETUP VARIABLES ---
SCRATCH_DIR="/homes/ks2218/la-proteina"
ENV_PATH="$SCRATCH_DIR/my_env"
CHECKPOINT_DIR="$SCRATCH_DIR/checkpoints_laproteina"

echo "Starting setup..."

# 2. Download ProteinMPNN weights
echo "Downloading ProteinMPNN weights..."
bash script_utils/download_pmpnn_weights.sh

# 3. Handle Conda Environment
# We use 'source' to ensure conda works inside this script
# Note: You might need to adjust the path to 'conda.sh' depending on your installation (e.g., ~/miniconda3/etc/...)
source $(conda info --base)/etc/profile.d/conda.sh

if conda info --envs | grep -q "^laproteina_env "; then
    echo "Environment exists. Updating..."
    conda env update --name "$laproteina_env" --file environment.yaml --prune
else
    echo "Creating new environment..."
    conda env create --name "$laproteina_env" --file environment.yaml

    
fi

conda activate "$ENV_PATH"

# 4. DOWNLOAD CHECKPOINTS TO SCRATCH (Answering your question)
echo "Setting up checkpoints in scratch space..."

# Create the directory in scratch if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# Download files directly into the scratch folder
curl -L 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/ld1_ucond_notri_512.ckpt/1.0/files?redirect=true&path=LD1_ucond_notri_512.ckpt' -o "$CHECKPOINT_DIR/LD1_ucond_notri_512.ckpt"
curl -L 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/ae1_ucond_512.ckpt/1.0/files?redirect=true&path=AE1_ucond_512.ckpt' -o "$CHECKPOINT_DIR/AE1_ucond_512.ckpt"

# OPTIONAL: Create a symlink so your python code finds the folder locally
# This makes "./checkpoints_laproteina" point to the scratch folder
ln -sfn "$CHECKPOINT_DIR" ./checkpoints_laproteina

# 5. Run the evaluation script
# I have changed this to 'bash' assuming it is a shell script. 
# If it is actually a python file, change 'bash' to 'python3' and rename the file to .py
echo "Running generation and evaluation..."
bash script_utils/gen_n_eval.sh

echo "Job Done!"