#!/usr/bin/env bash
# E125 follow-up: power up the L400 w32 cell to n=48 to test the (faint, n=16)
# "steering-to-AFDB-mean rescues codesign at L400" hint.
#
# Existing: afdb_target w32 L400 seeds 42-57 (n=16, in results/afdb_target_sweep).
# Add here: seeds 58-89 (32 more) -> 48 total steered.
# Unguided arm (n=48, identical recipe, nsteps=400, inference_ucond_notri_long):
#   seeds 42-57 = results/noise_aware_ensemble_sweep/codesign_unsteered_matched_seed.csv
#   seeds 58-89 = results/net_charge_target_L400_n64/unguided/codesign_guided.csv (E108)
# so NO new unguided generation is needed.
#
# Usage: nohup bash script_utils/run_afdb_L400_n48_extend.sh > nohup_afdb_L400_n48.out 2>&1 &
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
export IUPRED3_DIR=${IUPRED3_DIR:-/home/ks2218/iupred3}

ALL_SEEDS=$(seq -s ' ' 58 105)   # 48 fresh seeds (unguided exists for all of these via E108)
CFG=steering/config/sweep_afdb_target/afdb_target_w32.yaml
OUT=results/afdb_target_L400_n48/w32
DEVICE=cuda:0

# only generate seeds whose PDB is not already present (saves the few already done)
GEN_SEEDS=""
for s in $ALL_SEEDS; do
    [ -f "$OUT/guided/s${s}_n400.pdb" ] || GEN_SEEDS="$GEN_SEEDS $s"
done
if [ -n "$GEN_SEEDS" ]; then
    echo "[$(date -u +%FT%TZ)] generating afdb_target w32 L400 seeds:$GEN_SEEDS -> $OUT"
    "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long \
        --steering_config "$CFG" \
        --lengths 400 --seeds $GEN_SEEDS --nsteps 400 --skip_unguided \
        --output_dir "$OUT" --device $DEVICE
else
    echo "[$(date -u +%FT%TZ)] all seeds already generated; skip gen"
fi

echo "[$(date -u +%FT%TZ)] codesignability on the steered PDBs (seeds 58-105)"
OUT_BASE=results/afdb_target_L400_n48 "$PY" scripts/run_codesignability_sweep.py \
    --cfgs w32 --seeds $ALL_SEEDS --lengths 400

echo "[$(date -u +%FT%TZ)] === extension complete ==="
