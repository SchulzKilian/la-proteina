#!/bin/bash
#SBATCH -J precompute_latents
#SBATCH -A COMPUTERLAB-SL3-GPU       
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1            # Request 1 GPU for the AutoEncoder forward pass
#SBATCH --cpus-per-task=16      # Multiple cores help with globbing and file I/O
#SBATCH --mem=64G               # Latent precomputation can be memory intensive
#SBATCH --time=04:00:00         # Adjust based on dataset size
#SBATCH -p ampere               # Use the GPU partition
#SBATCH --output=slurm_precompute_%j.out

# 1. Load Modules (Matches your project's existing setup)
source $HOME/.bashrc
conda activate laproteina_env

# 2. Activate Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate laproteina_env

# 3. Set Up Paths
export PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd $PROJECT_DIR

# 4. Debug Info
echo "Running precomputation on host: $(hostname)"
nvidia-smi

# 5. Run the Script
# This assumes your python code is saved as precompute_latents.py in the root folder
python precompute_latents.py