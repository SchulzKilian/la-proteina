#!/bin/bash
#SBATCH -J thermstab
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_thermstab_%j.out
#
# Thermal-stability comparison (Tier 1 + Tier 2 / TemStaPro) of generated
# vs PDB sequences. Runs ProtT5-XL embeddings on a single A100, then the
# TemStaPro MLP heads across 9 temperature thresholds.
#
# Embedding cache lives at $TSP_EMB_DIR; reuse across reruns is automatic.
#
# Submit:
#   sbatch script_utils/run_thermal_stability.sh
#
# Local override (paths, length range) via env vars before sbatch:
#   GEN_FASTA=... REF_FASTA=... LENGTH_MIN=300 LENGTH_MAX=800 sbatch ...

source $HOME/.bashrc

# Use /home env (NOT /rds) to avoid Lustre Python-import hangs.
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

set -uo pipefail

cd /home/ks2218/la-proteina

# ── inputs ─────────────────────────────────────────────────────────────
GEN_FASTA="${GEN_FASTA:-results/generated_stratified_300_800/sequences.fasta}"
REF_FASTA="${REF_FASTA:-pdb_cluster_all_seqs.fasta}"
LENGTH_MIN="${LENGTH_MIN:-300}"
LENGTH_MAX="${LENGTH_MAX:-800}"
OUT_DIR="${OUT_DIR:-results/thermal_stability/stratified_vs_pdb}"

# Persistent locations (NOT /tmp — ProtT5 download is ~1.5 GB and
# embedding cache is hundreds of MB; keep them across job restarts).
TSP_DIR="${TSP_DIR:-/home/ks2218/TemStaPro}"
TSP_EMB_DIR="${TSP_EMB_DIR:-/home/ks2218/cache/temstapro_embeddings}"
export HF_HOME="${HF_HOME:-/home/ks2218/cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"

mkdir -p "$TSP_EMB_DIR" "$HF_HOME" "$OUT_DIR"

echo "── thermal stability run ──────────────────────────────────────────"
echo "  job id      : ${SLURM_JOB_ID:-local}"
echo "  node        : $(hostname)"
echo "  gpu         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo "  gen fasta   : $GEN_FASTA"
echo "  ref fasta   : $REF_FASTA"
echo "  length      : [$LENGTH_MIN, $LENGTH_MAX]"
echo "  out dir     : $OUT_DIR"
echo "  TemStaPro   : $TSP_DIR"
echo "  emb cache   : $TSP_EMB_DIR"
echo "  HF cache    : $HF_HOME"
echo "──────────────────────────────────────────────────────────────────"

python proteinfoundation/analysis/thermal_stability.py \
    --gen "$GEN_FASTA" \
    --ref "$REF_FASTA" \
    --out "$OUT_DIR" \
    --length-min "$LENGTH_MIN" \
    --length-max "$LENGTH_MAX" \
    --temstapro-dir "$TSP_DIR" \
    --temstapro-emb-dir "$TSP_EMB_DIR" \
    --python-exe "$LAPROTEINA_ENV/bin/python"

echo "── done ──────────────────────────────────────────────────────────"
ls -la "$OUT_DIR"
