#!/bin/bash
#SBATCH -J prepare_data_800
#SBATCH -A COMPUTERLAB-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH -p icelake
#SBATCH --output=slurm_prepare_data_%j.out

# CPU-only job: processes raw CIFs to .pt files.
# Icelake compute nodes have no outbound internet — run the download
# step from a login node first (raw CIFs land in data/pdb_train/raw/),
# then submit this job. It will see all CIFs present and skip straight
# to processing.
# Memory note: icelake caps at ~3370M/CPU, so 32 CPUs * 3370M ~= 105G max.

source $HOME/.bashrc
# Activate /home env via PATH prepend (NOT `conda activate`).
# /rds-based env hangs Python startup when any Lustre OST is evicted/disconn.
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

export DATA_PATH="/home/ks2218/la-proteina/data"
cd ~/la-proteina

echo "Running on host: $(hostname)"
echo "Using Python: $(which python)"
echo "CPUs: $(nproc)"

python prepare_data_800.py \
    dataset=pdb/pdb_train_ucond_800 \
    dataset.datamodule.num_workers=8
