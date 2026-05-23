#!/usr/bin/env bash
# Extend the unsteered codesign baseline from 30 → 48 PDBs (paired with the
# steered cells' 48). Waits for the fixt1 rerun chain to finish, then:
#   1. Generates unsteered PDBs for seeds 52-57 × L=300/400/500
#   2. Codesigns those 18 PDBs and appends to the baseline CSV
set -uo pipefail

cd /home/ks2218/la-proteina

export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release

LOG=/home/ks2218/la-proteina/results/sanity_unsteered_seed42_45/extend_52_57.log
mkdir -p /home/ks2218/la-proteina/results/sanity_unsteered_seed42_45/unguided

echo "=== START $(date -u +%FT%TZ) ===" | tee -a "$LOG"

# Step 1: wait for the rerun chain (PID 2020939) to finish so cuda:0 is free.
echo "[$(date -u +%FT%TZ)] waiting for fixt1 rerun chain (run_fixt1_full_replication.py) to exit..." | tee -a "$LOG"
while pgrep -f "run_fixt1_full_replication.py" >/dev/null; do
    sleep 30
done
echo "[$(date -u +%FT%TZ)] rerun chain gone; cuda:0 free." | tee -a "$LOG"

# Step 2: generate unsteered PDBs for seeds 52-57 (with --skip_guided, only
# unguided/ is written). default.yaml has steering.enabled: false, but we
# still pass it because the script requires the flag — output will be in
# results/sanity_unsteered_seed52_57/, we then move the unguided/ contents
# into the canonical sanity_unsteered_seed42_45/unguided/ dir.
TMP_OUT=/home/ks2218/la-proteina/results/sanity_unsteered_seed52_57
echo "[$(date -u +%FT%TZ)] generating unsteered PDBs for seeds 52-57 into $TMP_OUT" | tee -a "$LOG"
"$PY" -m steering.generate \
    --proteina_config inference_ucond_notri_long \
    --steering_config /home/ks2218/la-proteina/steering/config/default.yaml \
    --seeds 52 53 54 55 56 57 \
    --lengths 300 400 500 \
    --output_dir "$TMP_OUT" \
    --device cuda:0 \
    --nsteps 400 \
    --skip_guided 2>&1 | tee -a "$LOG"

# Step 3: stage PDBs in the canonical unguided dir
echo "[$(date -u +%FT%TZ)] staging PDBs into sanity_unsteered_seed42_45/unguided/" | tee -a "$LOG"
cp -n "$TMP_OUT"/unguided/*.pdb /home/ks2218/la-proteina/results/sanity_unsteered_seed42_45/unguided/ 2>&1 | tee -a "$LOG"
cp -n "$TMP_OUT"/unguided/*.pt  /home/ks2218/la-proteina/results/sanity_unsteered_seed42_45/unguided/ 2>&1 | tee -a "$LOG"

# Step 4: codesign the new 18 PDBs, append to baseline CSV
echo "[$(date -u +%FT%TZ)] codesigning seeds 52-57 and appending to baseline CSV" | tee -a "$LOG"
"$PY" scripts/codesign_unsteered_extend_52_57.py 2>&1 | tee -a "$LOG"

echo "=== END $(date -u +%FT%TZ) ===" | tee -a "$LOG"
