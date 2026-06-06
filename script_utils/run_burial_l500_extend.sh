#!/bin/bash
# After w48 frees GPU6/7, extend w64-L500 to n=96 (seeds 90-137), 2 GPUs. No GPU5.
set -o pipefail; cd /home/ks2218/la-proteina
export PATH=$HOME/.conda/envs/laproteina_env/bin:$PATH
PY=$HOME/.conda/envs/laproteina_env/bin/python
echo "[extend] waiting for w48 cells to free GPU6/7..."
until grep -q "CELL_DONE results/burial_camsol_probe/L500_w48_b0" logs/burial_l500_exp_gpu6.log 2>/dev/null \
   && grep -q "CELL_DONE results/burial_camsol_probe/L500_w48_b025" logs/burial_l500_exp_gpu7.log 2>/dev/null; do sleep 60; done
echo "[extend] w48 done; launching w64-L500 n->96 on GPU6+GPU7"
S="90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137"
CFG=steering/config/sweep_burial_camsol
CUDA=6 CFG=$CFG/camsol_w64_b0.yaml   OUT=results/burial_camsol_probe/L500_w64_b0   SEEDS="$S" bash script_utils/run_burial_cell.sh > logs/burial_l500_n96_b0.log 2>&1 &
P1=$!
CUDA=7 CFG=$CFG/camsol_w64_b025.yaml OUT=results/burial_camsol_probe/L500_w64_b025 SEEDS="$S" bash script_utils/run_burial_cell.sh > logs/burial_l500_n96_b025.log 2>&1 &
P2=$!
wait $P1; wait $P2
echo "[extend] n=96 folds done; verdict:"
$PY script_utils/run_burial_l500_eval.py
echo "=== L500_N96_DONE ==="
