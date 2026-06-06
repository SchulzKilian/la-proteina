#!/bin/bash
# Wait for both gen phases, fold codesign in parallel (GPU3 = w64 cells, GPU6 = w128
# cells), then compute the surfKD matched-band Fisher verdict from caches.
set -o pipefail
cd /home/ks2218/la-proteina
export PATH=$HOME/.conda/envs/laproteina_env/bin:$PATH
export MPNN_INPROCESS=1
PY=$HOME/.conda/envs/laproteina_env/bin/python
until grep -q "GEN_PHASE_w64extra_DONE" logs/burial_gen_w64.log 2>/dev/null \
   && grep -q "GEN_PHASE_w128_DONE"     logs/burial_gen_w128.log 2>/dev/null; do sleep 30; done
echo "[post] both gen phases done; folding in parallel"
CUDA_VISIBLE_DEVICES=3 $PY script_utils/run_burial_camsol_eval2.py fold \
    results/burial_camsol_probe/b0 results/burial_camsol_probe/b025 results/burial_camsol_probe/b010 \
    > logs/burial_fold_w64.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=6 $PY script_utils/run_burial_camsol_eval2.py fold \
    results/burial_camsol_probe/b0_w128 results/burial_camsol_probe/b025_w128 \
    > logs/burial_fold_w128.log 2>&1 &
P2=$!
wait $P1; wait $P2
echo "[post] folding done; computing verdict"
$PY script_utils/run_burial_camsol_eval2.py
echo "=== BURIAL_MORE2_DONE ==="
