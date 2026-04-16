#!/bin/bash
#SBATCH -J steer_part2
#SBATCH -A COMPUTERLAB-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_steerability_part2_%j.out

# Part 2: Property probes on proteins length 300-800.
# Requires processed_latents and data/properties.csv (from prep_properties.py).

source $HOME/.bashrc
conda activate laproteina_env

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
    echo "processed_latents already exists."
fi

# Prep the property file if not already done
if [ ! -f data/properties.csv ]; then
    echo "Creating properties.csv from developability_panel.csv..."
    python prep_properties.py
fi

echo "Running Part 2 on host: $(hostname)"
echo "Using Python: $(which python)"
echo "CPUs available: $(nproc)"

python -m src.part2_property_probes.run --config config/default.yaml
