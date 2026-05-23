#!/usr/bin/env bash
# Coord-channel sap_min sweep at w ∈ {32, 48, 64}.
# Same predictor (5-fold coord-NA ensemble) and protocol as the TANGO coord-only
# sweep — only objective + cell name differ. SAP is the structure-dependent
# property where CA-channel steering is most likely to pay off.
set -uo pipefail
cd /home/ks2218/la-proteina

PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
DEVICE=${1:-cuda:0}
WLIST=${2:-"32 48 64"}
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400 500"
NSTEPS=400
TREE_DIR=results/coords_na_eval

mkdir -p "$TREE_DIR"
echo "[$(date)] coord-channel SAP-min sweep: device=$DEVICE w-list=[$WLIST]"

for W in $WLIST; do
  CELL=coords_na_sap_coord_only_w${W}
  OUT_DIR=results/${CELL}
  CFG=steering/config/sweep_coords_na/sap_min_w${W}_coord_only.yaml

  echo ""
  echo "[$(date)] ============================================="
  echo "[$(date)] w=${W}  ::  generation (channel=bb_ca, SAP-min)"
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
      2>&1 | sed "s/^/[sap-ca-w${W} gen] /"

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
      2>&1 | sed "s/^/[sap-ca-w${W} prop] /"

  echo ""
  echo "[$(date)] w=${W}  ::  codesignability"
  OUT_BASE=results CUDA_VISIBLE_DEVICES=${DEVICE#cuda:} "$PY" scripts/run_codesignability_sweep.py \
      --cfgs "$CELL" \
      --seeds $SEEDS \
      --lengths $LENGTHS \
      2>&1 | sed "s/^/[sap-ca-w${W} codesign] /"

  echo "[$(date)] w=${W}  ::  done"
done

echo ""
echo "[$(date)] SAP-min coord-channel sweep complete."
