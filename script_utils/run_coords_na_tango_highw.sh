#!/usr/bin/env bash
# Coord-NA tango_min sweep at w ∈ {48, 64, 128} (w=32 already done).
# Sequential on a single GPU. Per cell: generation → property+AA audit → codesign.
# All cells live under results/coords_na_tango_w<W>/; symlinks in
# results/coords_na_eval/<cell>/ let steering_cost_audit.py discover them
# cleanly via its `*/guided` glob.
set -uo pipefail
cd /home/ks2218/la-proteina

PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
DEVICE=${1:-cuda:0}
WLIST=${2:-"48 64 128"}
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400 500"
NSTEPS=400
TREE_DIR=results/coords_na_eval

mkdir -p "$TREE_DIR"
echo "[$(date)] coord-NA high-w tango sweep: device=$DEVICE w-list=[$WLIST]"

for W in $WLIST; do
  CELL=coords_na_tango_w${W}
  OUT_DIR=results/${CELL}
  CFG=steering/config/sweep_coords_na/tango_min_w${W}.yaml

  echo ""
  echo "[$(date)] ============================================="
  echo "[$(date)] w=${W}  ::  generation"
  echo "[$(date)] ============================================="
  mkdir -p "$OUT_DIR"
  "$PY" -m steering.generate \
      --proteina_config inference_ucond_notri_long \
      --steering_config "$CFG" \
      --lengths $LENGTHS \
      --seeds $SEEDS \
      --nsteps $NSTEPS \
      --skip_unguided \
      --output_dir "$OUT_DIR" \
      --device "$DEVICE" \
      2>&1 | sed "s/^/[w${W} gen] /"

  N_PDB=$(ls "$OUT_DIR/guided"/*.pdb 2>/dev/null | wc -l)
  echo "[$(date)] w=${W}  ::  $N_PDB PDBs ready"
  if [[ "$N_PDB" -eq 0 ]]; then
    echo "[$(date)] w=${W}  ::  no PDBs, aborting sweep"; exit 2
  fi

  echo ""
  echo "[$(date)] w=${W}  ::  property + AA audit"
  ln -snf "../${CELL}" "$TREE_DIR/${CELL}"
  "$PY" script_utils/steering_cost_audit.py \
      --tree "$TREE_DIR" \
      --evals property,aa \
      --seeds $SEEDS \
      --lengths $LENGTHS \
      2>&1 | sed "s/^/[w${W} prop] /"

  echo ""
  echo "[$(date)] w=${W}  ::  codesignability"
  OUT_BASE=results CUDA_VISIBLE_DEVICES=${DEVICE#cuda:} "$PY" scripts/run_codesignability_sweep.py \
      --cfgs "$CELL" \
      --seeds $SEEDS \
      --lengths $LENGTHS \
      2>&1 | sed "s/^/[w${W} codesign] /"

  echo "[$(date)] w=${W}  ::  done"
done

echo ""
echo "[$(date)] sweep complete."
