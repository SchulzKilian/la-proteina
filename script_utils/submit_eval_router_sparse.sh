#!/bin/bash
#SBATCH -J eval_router_sparse
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2:30:00
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_eval_router_sparse_%j.out

# Move-2 paired sampling + scRMSD eval of the router-sparse trunk vs the
# canonical dense baseline (E019).
#
# Submit modes:
#   1. Intermediate-checkpoint probe (N=6 per length):
#        sbatch --export=LABEL=probe_step1500,SPARSE_CKPT=<path>,N=6 \
#            script_utils/submit_eval_router_sparse.sh
#   2. Full eval (N=12 per length):
#        sbatch --export=LABEL=full_step5000,SPARSE_CKPT=<path>,N=12 \
#            script_utils/submit_eval_router_sparse.sh
#
# Both arms always run; if you need baseline-only or sparse-only, pass
# EXTRA_ARGS="--baseline_only" / "--sparse_only".

# Source bashrc FIRST (E065 fix), then `set` flags.
source $HOME/.bashrc
set -uo pipefail

export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

cd $HOME/la-proteina

# Required env: LABEL, SPARSE_CKPT. Optional: N (default 6), LENGTHS (default 100,200),
# SEED (default 5), EXTRA_ARGS (default empty), MPNN_SEQS (default 8).
: "${LABEL:?ERROR: pass LABEL=<run_tag>}"
: "${SPARSE_CKPT:?ERROR: pass SPARSE_CKPT=<path/to/last.ckpt or best_val_*.ckpt>}"
: "${N:=6}"
: "${LENGTHS:=100,200}"
: "${SEED:=5}"
: "${EXTRA_ARGS:=}"
: "${MPNN_SEQS:=8}"

# ProteinMPNN subprocess: see MEMORY.md feedback_export_python_exec — without
# this, scRMSD subprocesses use bare `python` and silently return -1.0.
export PYTHON_EXEC="$LAPROTEINA_ENV/bin/python"

echo "[+] node:        $(hostname)"
echo "[+] python:      $(which python)"
echo "[+] LABEL:       $LABEL"
echo "[+] SPARSE_CKPT: $SPARSE_CKPT"
echo "[+] N samples/L: $N"
echo "[+] LENGTHS:     $LENGTHS"
echo "[+] SEED:        $SEED"
echo "[+] MPNN_SEQS:   $MPNN_SEQS"
echo "[+] EXTRA_ARGS:  $EXTRA_ARGS"
echo "[+] PYTHON_EXEC: $PYTHON_EXEC"

t_start=$(date +%s)

# ---- Phase 1: sample (paired noise) ----
echo "================================================================================"
echo "[Phase 1] sampling N=$N per L=$LENGTHS at nsteps=400 ..."
echo "================================================================================"
GEN_LOG="slurm_eval_router_sparse_${SLURM_JOB_ID:-$$}.gen.log"
python script_utils/sample_router_sparse_paired.py \
    --label "$LABEL" \
    --sparse_ckpt "$SPARSE_CKPT" \
    --n_samples "$N" \
    --lengths "$LENGTHS" \
    --seed "$SEED" \
    --nsteps 400 \
    $EXTRA_ARGS \
    2>&1 | tee "$GEN_LOG"
echo "[+] sampling done after $(( $(date +%s) - t_start ))s"

# ---- Phase 2: scRMSD ----
echo "================================================================================"
echo "[Phase 2] scRMSD evaluation (ProteinMPNN + ESMFold) ..."
echo "================================================================================"
EVAL_LOG="slurm_eval_router_sparse_${SLURM_JOB_ID:-$$}.eval.log"
python script_utils/eval_router_sparse_paired.py \
    --label "$LABEL" \
    --n_samples "$N" \
    --lengths "$LENGTHS" \
    --num_seq_per_target "$MPNN_SEQS" \
    2>&1 | tee "$EVAL_LOG"

echo "[+] all phases done in $(( $(date +%s) - t_start ))s"
echo "[+] CSV at: results/sparse_trunk_move2/$LABEL/results_scrmsd.csv"
