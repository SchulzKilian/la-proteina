#!/bin/bash
#SBATCH -p sapphire
#SBATCH -A COMPUTERLAB-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --job-name=prune_latents
#SBATCH --output=logs/prune_%j.out

source $HOME/.bashrc

# 2. Activate the environment
# Activate /home env via PATH prepend (NOT `conda activate`).
# /rds-based env hangs Python startup when any Lustre OST is evicted/disconn.
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

# 2. Run the pruning script
# Note: Ensure the NUM_WORKERS in the python script matches --cpus-per-task
python /home/ks2218/la-proteina/clean_latents.py