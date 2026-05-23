#!/usr/bin/env bash
# Queue the coord-channel sweep behind the latent-channel sweep currently
# running. Steps:
#   1. Wait for the latent-channel driver PID to exit.
#   2. Run a 1-PDB smoke for channel=bb_ca to confirm the new dispatch path
#      runs end-to-end before paying for the 4×48-PDB sweep.
#   3. Launch the full coord-only sweep via run_coords_na_tango_coordonly.sh.
set -uo pipefail
cd /home/ks2218/la-proteina

PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
WAIT_PID=${1:?"usage: $0 <latent-sweep-PID> [cuda:N=cuda:0]"}
DEVICE=${2:-cuda:0}

echo "[$(date)] waiting for latent-channel sweep PID $WAIT_PID to exit ..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[$(date)] PID $WAIT_PID exited. running coord-channel smoke ..."

mkdir -p results/coords_na_coord_only_smoke
SMOKE_LOG=results/coords_na_coord_only_smoke/runlog.log
"$PY" -m steering.generate \
    --proteina_config inference_ucond_notri_long \
    --steering_config steering/config/sweep_coords_na/tango_min_w32_coord_only.yaml \
    --lengths 300 \
    --seeds 42 \
    --nsteps 400 \
    --skip_unguided \
    --output_dir results/coords_na_coord_only_smoke \
    --device "$DEVICE" \
    > "$SMOKE_LOG" 2>&1

if ! ls results/coords_na_coord_only_smoke/guided/*.pdb >/dev/null 2>&1; then
  echo "[$(date)] SMOKE FAILED — no PDB written. Inspect $SMOKE_LOG. Aborting."
  exit 2
fi
echo "[$(date)] smoke ok. launching full coord-channel sweep."

bash script_utils/run_coords_na_tango_coordonly.sh "$DEVICE" "32 48 64"
echo "[$(date)] coord-channel sweep done."
