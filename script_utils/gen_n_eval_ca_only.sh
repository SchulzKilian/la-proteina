#!/bin/bash
#SBATCH -J eval_ca_only
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00

source $HOME/.bashrc
# Activate /home env via PATH prepend (NOT `conda activate`).
# /rds-based env hangs Python startup when any Lustre OST is evicted/disconn.
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

# gen_n_eval_ca_only.sh
# Generate structures and run designability evaluation for a CA-only 70M checkpoint.
#
# Requires: GPU with >=16GB VRAM (ESMFold). Do NOT run on a login node.
# Estimated runtime: ~10 min generation + ~75 min evaluation = ~1.5 hours total on A100.
# (140 PDBs x 8 ESMFold calls each, 50-200 residue proteins)
#
# Usage:
#   bash script_utils/gen_n_eval_ca_only.sh               # auto-finds latest last.ckpt
#   bash script_utils/gen_n_eval_ca_only.sh --ckpt ./store/ca_only_diffusion_70M/1234567890/last.ckpt

set -euo pipefail

CONFIG_NAME="inference_ucond_ca_only_70M"
STORE_DIR="./store/ca_only_diffusion_70M"
PMPNN_SCRIPT="./ProteinMPNN/protein_mpnn_run.py"

# ── Sanity checks ──────────────────────────────────────────────────────────
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: No CUDA GPU available. This script requires a GPU (ESMFold needs >=16GB VRAM)."
    echo "       Submit as a SLURM job, e.g.:"
    echo "         sbatch --partition=ampere --gres=gpu:1 --time=5:00:00 \\"
    echo "                --wrap=\"bash script_utils/gen_n_eval_ca_only.sh\""
    exit 1
fi

if [[ ! -f "$PMPNN_SCRIPT" ]]; then
    echo "ERROR: ProteinMPNN weights not found at $PMPNN_SCRIPT"
    echo "       Run once: bash script_utils/download_pmpnn_weights.sh"
    exit 1
fi

# ── Parse args ─────────────────────────────────────────────────────────────
CKPT_PATH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt) CKPT_PATH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Auto-detect checkpoint if not provided ─────────────────────────────────
if [[ -z "$CKPT_PATH" ]]; then
    CKPT_PATH=$(find "$STORE_DIR" -name "last.ckpt" -printf "%T@ %p\n" 2>/dev/null \
                | sort -n | tail -1 | awk '{print $2}')
    if [[ -z "$CKPT_PATH" ]]; then
        echo "ERROR: No last.ckpt found under $STORE_DIR"
        echo "       Pass one explicitly: --ckpt <path>"
        exit 1
    fi
    echo "Auto-detected checkpoint: $CKPT_PATH"
fi

if [[ ! -f "$CKPT_PATH" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

CKPT_DIR=$(dirname "$CKPT_PATH")
CKPT_NAME=$(basename "$CKPT_PATH")

echo "============================================"
echo "Config     : $CONFIG_NAME"
echo "Checkpoint : $CKPT_PATH"
echo "GPU        : $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "============================================"

# ── Clear previous inference results for this config ──────────────────────
INFER_DIR="./inference/${CONFIG_NAME}"
if [[ -d "$INFER_DIR" ]]; then
    echo "Removing old inference dir: $INFER_DIR"
    rm -rf "$INFER_DIR"
fi

# ── Generate ───────────────────────────────────────────────────────────────
echo ""
echo ">>> Generating samples..."
python proteinfoundation/generate.py \
    --config_name "$CONFIG_NAME" \
    ckpt_path="$CKPT_DIR" \
    ckpt_name="$CKPT_NAME"

# ── Evaluate ───────────────────────────────────────────────────────────────
echo ""
echo ">>> Evaluating samples (ProteinMPNN + ESMFold, ~75 min on A100)..."
python proteinfoundation/evaluate.py \
    --config_name "$CONFIG_NAME" \
    ckpt_path="$CKPT_DIR" \
    ckpt_name="$CKPT_NAME"

echo ""
echo "Done. Results CSV: ./results_${CONFIG_NAME}_0.csv"
