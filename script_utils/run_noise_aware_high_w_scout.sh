#!/usr/bin/env bash
# Under-steering vs free-direction scout: extend the noise-aware ensemble sweep
# to w ∈ {32, 64, 128} for one direction (camsol_max OR tango_min), matching
# the original 16 seeds × 3 lengths × nsteps=400 protocol. Paired-by-seed with
# the existing noise_aware_ensemble_sweep (seeds 42-57) so within-noise
# w-scaling can be read directly.
#
# One-direction-per-GPU design: launch twice (once per direction) on separate
# GPUs to run in parallel; or once for single-GPU sequential.
#
# Usage:
#   nohup bash script_utils/run_noise_aware_high_w_scout.sh camsol_max cuda:2 \
#         > nohup_high_w_camsol.out 2>&1 &
#   nohup bash script_utils/run_noise_aware_high_w_scout.sh tango_min  cuda:3 \
#         > nohup_high_w_tango.out  2>&1 &
set -uo pipefail
cd /home/ks2218/la-proteina

DIRECTION=${1:?"usage: $0 <camsol_max|tango_min> [cuda:N=cuda:0] [\"w1 w2 ...\"=\"32 64 128\"]"}
DEVICE=${2:-cuda:0}
WLEVELS_STR=${3:-"32 64 128"}
read -ra WLEVELS <<< "$WLEVELS_STR"

PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400 500"
NSTEPS=400
OUT_ROOT=results/noise_aware_high_w_scout

mkdir -p "$OUT_ROOT"
echo "[$(date)] high-w scout: direction=$DIRECTION device=$DEVICE"
echo "3 w-levels × 16 seeds × 3 lengths = 144 PDBs."
echo

for w in "${WLEVELS[@]}"; do
  cfg="${DIRECTION}_w${w}"
  out="$OUT_ROOT/$cfg"
  echo "[$(date)] [$DEVICE] starting $cfg"
  "$PY" -m steering.generate \
    --proteina_config inference_ucond_notri_long \
    --steering_config "steering/config/sweep_noise_aware_high_w/${cfg}.yaml" \
    --lengths $LENGTHS \
    --seeds $SEEDS \
    --nsteps $NSTEPS \
    --skip_unguided \
    --output_dir "$out" \
    --device "$DEVICE"
  echo "[$(date)] [$DEVICE] finished $cfg"
done

echo
echo "[$(date)] high-w scout for $DIRECTION on $DEVICE: complete."
