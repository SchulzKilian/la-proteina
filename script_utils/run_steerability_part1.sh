#!/bin/bash
#SBATCH -J steer_part1
#SBATCH -A COMPUTERLAB-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_steerability_part1_%j.out

# Part 1: Latent geometry analysis on proteins length 300-800.
# Requires processed_latents to exist (restores from tar shards if needed).

source $HOME/.bashrc
# Activate /home env via PATH prepend (NOT `conda activate`).
# /rds-based env hangs Python startup when any Lustre OST is evicted/disconn.
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

cd ~/la-proteina/laproteina_steerability

# Restore processed_latents from tar shards if not present
LATENT_DIR="/rds/user/ks2218/hpc-work/processed_latents"
SHARD_DIR="/rds/user/ks2218/hpc-work/latent_shards"

if [ ! -d "$LATENT_DIR" ] || [ -z "$(ls -A "$LATENT_DIR" 2>/dev/null)" ]; then
    echo "Restoring processed_latents from tar shards..."
    mkdir -p "$LATENT_DIR"
    ls "$SHARD_DIR"/*.tar | xargs -P 32 -I{} tar -xf {} -C "$LATENT_DIR"
    echo "Restored $(find "$LATENT_DIR" -name '*.pt' | wc -l) latent files."
else
    echo "processed_latents already exists with $(find "$LATENT_DIR" -name '*.pt' | wc -l) files."
fi

echo "Running Part 1 on host: $(hostname)"
echo "Using Python: $(which python)"

python -m src.part1_latent_geometry.run --config config/default.yaml
