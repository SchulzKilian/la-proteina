#!/bin/bash
# One generation phase of the burial confirmation run, on ONE gpu. Resume-safe.
#   PHASE=w64extra : w64 b0+b025 at seeds 58-73 (appended -> n=64/cell)
#   PHASE=w128     : w128 b0+b025 at seeds 42-57 (higher-w test)
set -o pipefail
cd /home/ks2218/la-proteina
export PATH=$HOME/.conda/envs/laproteina_env/bin:$PATH
export MPNN_INPROCESS=1
PY=$HOME/.conda/envs/laproteina_env/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA:?set CUDA}
LENGTHS="300 400"; ROOT=results/burial_camsol_probe
case "${PHASE:?set PHASE}" in
  w64extra) CFGW=w64;  SEEDS="58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73"; SUFFIX="" ;;
  w128)     CFGW=w128; SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"; SUFFIX="_w128" ;;
  *) echo "bad PHASE"; exit 1 ;;
esac
echo "[gen $PHASE] gpu=$CUDA_VISIBLE_DEVICES seeds=$SEEDS"
for tag in b0 b025; do
  echo "[$(date -u +%FT%TZ)] $PHASE $tag"
  "$PY" -m steering.generate --proteina_config inference_ucond_notri_long \
      --steering_config steering/config/sweep_burial_camsol/camsol_${CFGW}_${tag}.yaml \
      --lengths $LENGTHS --seeds $SEEDS --nsteps 400 --skip_unguided --resume \
      --output_dir "$ROOT/${tag}${SUFFIX}" --device cuda:0
done
echo "[$(date -u +%FT%TZ)] === GEN_PHASE_${PHASE}_DONE ==="
