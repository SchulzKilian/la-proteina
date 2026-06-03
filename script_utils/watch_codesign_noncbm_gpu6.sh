#!/usr/bin/env bash
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"; LENGTHS="300 400 500"
CELLS="rama:128 aaprior:128 rama:64 aaprior:64"
for spec in $CELLS; do
  arm=${spec%%:*}; w=${spec##*:}; cfg="tango_min_w${w}"; dir="results/throttle_noncbm/$arm/$cfg"
  [ -f "$dir/codesign_guided.csv" ] && { echo "[$(date -u +%FT%TZ)] $arm/$cfg codesign exists; skip"; continue; }
  while true; do
    n=$(ls "$dir/guided"/*.pdb 2>/dev/null | wc -l); [ "$n" -ge 48 ] && break
    grep -q "non-CBM throttle sweep complete" nohup_throttle_noncbm.log 2>/dev/null && break
    sleep 30
  done
  n=$(ls "$dir/guided"/*.pdb 2>/dev/null | wc -l)
  [ "$n" -lt 48 ] && { echo "[$(date -u +%FT%TZ)] $arm/$cfg only $n; skip"; continue; }
  echo "[$(date -u +%FT%TZ)] === codesign $arm/$cfg on GPU6 ($n pdbs) ==="
  CUDA_VISIBLE_DEVICES=6 OUT_BASE="results/throttle_noncbm/$arm" \
    "$PY" scripts/run_codesignability_sweep.py --seeds $SEEDS --lengths $LENGTHS --cfgs "$cfg" \
    && echo "[$(date -u +%FT%TZ)] $arm/$cfg codesign DONE" || echo "[$(date -u +%FT%TZ)] $arm/$cfg codesign FAILED"
done
echo "[$(date -u +%FT%TZ)] === noncbm codesign watcher complete ==="
