#!/usr/bin/env bash
# Parallel copy of the conda env from slow NFS to /dev/shm. NFS small-file reads are
# metadata-latency-bound; concurrency hides the per-file round-trips. File-level fan-out
# (not dir-level) so the big torch package is also parallelized internally.
set -uo pipefail
SRC=/home/ks2218/.conda/envs/laproteina_env
DST=/dev/shm/lpenv/laproteina_env
rm -rf /dev/shm/lpenv
mkdir -p "$DST"
echo "[$(date -u +%FT%TZ)] listing dirs..."
( cd "$SRC" && find . -type d ) | ( cd "$DST" && xargs -P 32 -I{} mkdir -p "{}" )
echo "[$(date -u +%FT%TZ)] dirs done; copying files (file-level, -P 64)..."
( cd "$SRC" && find . -type f -o -type l ) > /tmp/parcopy_files.txt
wc -l /tmp/parcopy_files.txt
cat /tmp/parcopy_files.txt | xargs -P 64 -I{} cp -a "$SRC/{}" "$DST/{}" 2>/dev/null
echo "PARCOPY_DONE $(date -u +%FT%TZ)"
du -sh "$DST"