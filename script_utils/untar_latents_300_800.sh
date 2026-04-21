#!/bin/bash
#SBATCH -J untar_300_800
#SBATCH -A COMPUTERLAB-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH -o slurm_untar_300_800_%j.out

# Do NOT use set -e (SLURM TaskProlog mkdir trips it).
set -uo pipefail

SRC_TAR=/rds/user/ks2218/hpc-work/processed_latents.tar
EXTRACT_ROOT=/rds/user/ks2218/hpc-work
OUT_DIR=/rds/user/ks2218/hpc-work/processed_latents
LINK=/home/ks2218/la-proteina/data/pdb_train/processed_latents_300_800
WORKERS=16

echo "Node: $(hostname)"
echo "Source: $SRC_TAR ($(du -sh $SRC_TAR 2>/dev/null | cut -f1))"

if [ -d "$OUT_DIR" ] && [ "$(ls -A "$OUT_DIR" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "ERROR: $OUT_DIR non-empty. Delete it first."
    exit 2
fi

echo "Pre-extract quota:"
lfs quota -u ks2218 /rds/user/ks2218 2>&1 | sed -n '2,4p'

# Enumerate unique top-level prefixes inside the archive so we can parallelise.
# Archive structure: processed_latents/<2char_prefix>/<pdb_chain>.pt
PREFIX_LIST=$(mktemp)
echo "Listing prefixes at $(date) ..."
tar -tf "$SRC_TAR" 2>/dev/null | awk -F'/' 'NF>=3 && $2 != "" {print $2}' | sort -u > "$PREFIX_LIST"
N_PREFIXES=$(wc -l < "$PREFIX_LIST")
echo "Found $N_PREFIXES prefixes"

mkdir -p "$OUT_DIR"

echo "Starting parallel extraction with $WORKERS workers at $(date)..."
# Each worker scans the whole tar but only extracts files under its prefix —
# trades extra sequential reads for parallel Lustre metadata ops (mkdir + create).
cat "$PREFIX_LIST" | xargs -P $WORKERS -I {} tar -xf "$SRC_TAR" -C "$EXTRACT_ROOT" --wildcards "processed_latents/{}/*"
rc=$?
echo "Extraction finished at $(date) (rc=$rc)"

FILE_COUNT=$(find "$OUT_DIR" -name '*.pt' 2>/dev/null | wc -l)
DIR_COUNT=$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "File count: $FILE_COUNT"
echo "Shard-dir count: $DIR_COUNT"

echo "Post-extract quota:"
lfs quota -u ks2218 /rds/user/ks2218 2>&1 | sed -n '2,4p'

rm -f "$PREFIX_LIST"

# Refresh symlink used by laproteina_steerability config
ln -sfn "$OUT_DIR" "$LINK"
echo "Symlink: $(ls -ld "$LINK")"

exit $rc
