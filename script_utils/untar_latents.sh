#!/bin/bash
#SBATCH -J untar_latents
#SBATCH -A COMPUTERLAB-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH -o slurm_untar_latents_%j.out

# Do NOT use set -e (SLURM TaskProlog mkdir trips it).
set -uo pipefail

SHARD_DIR=/rds/user/ks2218/hpc-work/latent_shards
OUT_DIR=/rds/user/ks2218/hpc-work/processed_latents
LINK=/home/ks2218/la-proteina/data/pdb_train/processed_latents_300_800

echo "Node: $(hostname)"
echo "Shards: $(ls $SHARD_DIR/*.tar 2>/dev/null | wc -l) tars"
du -sh "$SHARD_DIR" 2>/dev/null

mkdir -p "$OUT_DIR"

echo "Starting parallel untar at $(date)"
ls "$SHARD_DIR"/*.tar | xargs -P 32 -I{} tar -xf {} -C "$OUT_DIR"
rc=$?
echo "Untar finished at $(date) (rc=$rc)"

echo "File count: $(find "$OUT_DIR" -mindepth 2 -type f | wc -l)"
echo "Shard-dir count: $(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"

# Refresh symlink used by laproteina_steerability config
ln -sfn "$OUT_DIR" "$LINK"
echo "Symlink: $(ls -ld "$LINK")"

exit $rc
