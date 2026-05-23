#!/usr/bin/env bash
# Post-gen evaluation chain for the coord-NA tango_min_w32 cell.
# Waits for the active generation PID to exit, then runs property+AA audit
# (CPU-side) followed by codesignability (GPU-side, cuda:0).
#
# Idempotent: codesign rows are appended on protein_id, property eval
# re-writes properties_guided.csv inside the cell.
set -uo pipefail
cd /home/ks2218/la-proteina

PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
GEN_PID=${1:-}
CELL_DIR=results/coords_na_tango_w32
TREE_DIR=results/coords_na_eval

if [[ -n "$GEN_PID" ]]; then
  echo "[$(date)] waiting for gen PID $GEN_PID to exit ..."
  while kill -0 "$GEN_PID" 2>/dev/null; do sleep 30; done
  echo "[$(date)] gen PID $GEN_PID exited."
fi

if ! ls "$CELL_DIR/guided"/*.pdb >/dev/null 2>&1; then
  echo "[$(date)] ERROR: no PDBs at $CELL_DIR/guided/"; exit 2
fi
N_PDB=$(ls "$CELL_DIR/guided"/*.pdb | wc -l)
echo "[$(date)] $N_PDB guided PDBs ready."

# --- 1) Property + AA audit (CPU-bound)
mkdir -p "$TREE_DIR"
ln -snf "../coords_na_tango_w32" "$TREE_DIR/coords_na_tango_w32"

echo ""
echo "[$(date)] === property + AA audit ==="
"$PY" script_utils/steering_cost_audit.py \
    --tree "$TREE_DIR" \
    --evals property,aa \
    --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 \
    --lengths 300 400 500 \
    2>&1 | sed 's/^/[prop] /'

# --- 2) Codesignability (GPU-bound, cuda:0)
echo ""
echo "[$(date)] === codesignability sweep (cuda:0) ==="
OUT_BASE=results CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run_codesignability_sweep.py \
    --cfgs coords_na_tango_w32 \
    --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 \
    --lengths 300 400 500 \
    2>&1 | sed 's/^/[codesign] /'

echo ""
echo "[$(date)] === post-gen eval chain complete ==="
echo "Property:  $CELL_DIR/properties_guided.csv"
echo "AA audit:  $TREE_DIR/aa_collapse_summary.csv  (and per-cell)"
echo "Codesign:  $CELL_DIR/codesign_guided.csv"
