#!/bin/bash
#SBATCH --job-name=laproteina_adaptive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=5:00:00
#SBATCH --partition=ampere
#SBATCH --output=logs/slurm/%j.out

# 1. Establish the "Anchor" (Project Root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 2. Environment Activation
# Using the ENV_NAME defined in your snippet
source /usr/local/software/anaconda/3/etc/profile.d/conda.sh
conda activate laproteina_env

# 3. HPC Module Loading (Required for CSD3 Ampere nodes)
module purge
module load rhel8/default-amp
module load cuda/11.8

# 4. Performance & Memory Tuning
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_COMPILE_DISABLE=1

# 5. Execution
# We use the adaptive script to ensure the correct AE checkpoint is selected
# based on the 200-residue limit.
bash script_utils/full_training_adaptive.sh \
    dataset.datamodule.dataselector.min_length=100 \
    dataset.datamodule.dataselector.max_length=200 \
    dataset.datamodule.max_size=200 \
    dataset.datamodule.batch_size=12 \
