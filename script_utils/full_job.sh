#!/bin/bash
#SBATCH -J laproteina_full
#SBATCH -A CHANGEME-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00        # Note: Full training takes much longer!
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load miniconda/3

source $(conda info --base)/etc/profile.d/conda.sh
conda activate laproteina_env

# 1. Wake up the shared storage
echo "Mounting shared PDB..."
ls /datasets/public/AlphaFold/data/pdb_mmcif > /dev/null

# 2. Run with Hydra Overrides
echo "Starting Full Scale Training..."
python proteinfoundation/train.py \
    dataset=pdb/pdb_train_ucond \
    dataset.data_dir=/datasets/public/AlphaFold/data \
    dataset.fraction=0.5 \
    nn=local_latents_score_nn_160M \
    hydra.run.dir="logs/training/full_$(date +%Y%m%d_%H%M%S)"