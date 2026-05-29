#!/bin/bash
# Queue w1/w2 geometric look-ahead arms on the SAME GPU as the running
# designability job (no 2nd GPU): wait for PID 908731 -> generate w1,w2
# (both targets x baseline+prop = 8 arms) -> fold them (self-consistency).
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
export CUDA_VISIBLE_DEVICES=GPU-fee3d1d9-0d9a-b6af-3cdf-54ffc45086a6   # GPU 1

WAIT_PID=908731
echo "[chain] $(date) waiting for designability PID ${WAIT_PID} to finish..."
while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
echo "[chain] $(date) PID ${WAIT_PID} done. Generating w1/w2 arms."

${PY} -m steering.run_geom_lookahead_sweep --device cuda:0 --lambdas 1 2
echo "[chain] $(date) generation complete. Starting designability."

CFGS="rg_baseline_w1 rg_prop_w1 rg_baseline_w2 rg_prop_w2 contact_order_baseline_w1 contact_order_prop_w1 contact_order_baseline_w2 contact_order_prop_w2"
OUT_BASE=results/geom_lookahead_sweep ${PY} script_utils/run_scrmsd_steering.py \
    --cfgs ${CFGS} --lengths 300 400 --seeds 42 43 44 45 46 47
echo "[chain] $(date) w1/w2 designability complete."
