#!/bin/bash
#!
#! Slurm job script for LaProteina
#!
#SBATCH -J laproteina_train
#SBATCH -A CHANGEME-GPU         # <--- CHANGE THIS to your project account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1            # Request 1 A100 GPU
#SBATCH --cpus-per-task=16      # Request CPU cores for data loading
#SBATCH --time=12:00:00
#SBATCH -p ampere               # Partition for A100s
#SBATCH --output=slurm-%j.out

# 1. Load Modules
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load miniconda/3

# 2. Activate Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate laproteina_env

# 3. Set Up Paths
# We set PROJECT_DIR to the directory where you submitted the job
export PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd $PROJECT_DIR

# CRITICAL: This variable tells the config where to find the data
# We match the path you used during the setup on the login node
export DATA_PATH="$PROJECT_DIR/data"

# 4. Debug Info
echo "Running on host: $(hostname)"
echo "Data Path: $DATA_PATH"
nvidia-smi

# 5. Run Training
# The data is already in $DATA_PATH, so the script will skip download and go straight to training.
python proteinfoundation/train.py \
    dataset=pdb/pdb_train_ucond \
    nn=local_latents_score_nn_160M \
    hydra.run.dir="logs/training/$(date +%Y%m%d_%H%M%S)"