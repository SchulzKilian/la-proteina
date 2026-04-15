#!/bin/bash
#SBATCH -J prepare_data_800
#SBATCH -A COMPUTERLAB-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -p icelake
#SBATCH --output=slurm_prepare_data_%j.out

# CPU-only job: downloads raw CIFs from PDB, processes to .pt files.
# No GPU needed. Network access required for PDB downloads.

source $HOME/.bashrc
conda activate laproteina_env

export DATA_PATH="/home/ks2218/la-proteina/data"
cd ~/la-proteina

echo "Running on host: $(hostname)"
echo "Using Python: $(which python)"
echo "CPUs: $(nproc)"

python prepare_data_800.py \
    dataset=pdb/pdb_train_ucond_800
