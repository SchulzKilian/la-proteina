#!/usr/bin/env bash
# Concurrent codesign (ESMFold on GPU6) for throttle-sweep cells as they finish
# generating on GPU5. Per-cell, resume-safe. Pipelines the slow ESMFold leg with
# generation so it doesn't all land at the end.
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400 500"
# generation order
CELLS="nothrottle:128 rama:128 aaprior:128 nothrottle:64 rama:64 aaprior:64"
SWEEP_LOG=nohup_throttle_sweep.log
for spec in $CELLS; do
  arm=${spec%%:*}; w=${spec##*:}; cfg="tango_min_w${w}"
  dir="results/throttle_sweep/$arm/$cfg"
  csv="$dir/codesign_guided.csv"
  if [ -f "$csv" ]; then echo "[$(date -u +%FT%TZ)] $arm/$cfg codesign exists; skip"; continue; fi
  # wait for the cell's 48 PDBs (or the sweep to finish, in case a cell was skipped)
  while true; do
    n=$(ls "$dir/guided"/*.pdb 2>/dev/null | wc -l)
    [ "$n" -ge 48 ] && break
    grep -q "throttle full sweep complete" "$SWEEP_LOG" 2>/dev/null && break
    sleep 30
  done
  n=$(ls "$dir/guided"/*.pdb 2>/dev/null | wc -l)
  if [ "$n" -lt 48 ]; then echo "[$(date -u +%FT%TZ)] $arm/$cfg only $n pdbs; skip"; continue; fi
  echo "[$(date -u +%FT%TZ)] === codesign $arm/$cfg on GPU6 ($n pdbs) ==="
  CUDA_VISIBLE_DEVICES=6 OUT_BASE="results/throttle_sweep/$arm" \
    "$PY" scripts/run_codesignability_sweep.py --seeds $SEEDS --lengths $LENGTHS --cfgs "$cfg" \
    && echo "[$(date -u +%FT%TZ)] $arm/$cfg codesign DONE" \
    || echo "[$(date -u +%FT%TZ)] $arm/$cfg codesign FAILED"
done
echo "[$(date -u +%FT%TZ)] === concurrent codesign watcher complete ==="
