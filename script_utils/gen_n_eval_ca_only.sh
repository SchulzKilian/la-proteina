#!/bin/bash
#SBATCH -J eval_ca_only
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_eval_v2_quick_%j.out

source $HOME/.bashrc
# Activate /home env via PATH prepend (NOT `conda activate`).
# /rds-based env hangs Python startup when any Lustre OST is evicted/disconn.
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

# gen_n_eval_ca_only.sh
# Generate structures and run designability evaluation for the 160M CA-only v2
# checkpoint (trained under cosine_with_warmup + wd=0.1, best at step 2078).
# The "quick" config reduces the workload to fit in a 1h interactive session:
# 4 lengths × 10 samples × 200 ODE steps (vs the old default of 7 × 20 × 400).
#
# Requires: GPU with >=16GB VRAM (ESMFold). Do NOT run on a login node.
# Estimated runtime: ~15 min generation + ~25 min evaluation = ~40 min total on A100.
#
# Usage:
#   bash script_utils/gen_n_eval_ca_only.sh               # uses ckpt baked into the config
#   bash script_utils/gen_n_eval_ca_only.sh --ckpt <path> # override with any specific ckpt

set -uo pipefail
# NOTE: deliberately NOT `set -e`. Cambridge HPC's SLURM TaskProlog runs an
# `mkdir /var/spool/slurm/slurmd/logs` that fails with permission-denied; with
# `set -e` that aborts the whole job before user code runs. See CLAUDE.md.

CONFIG_NAME="inference_ucond_notri_ca_only_v2_quick"
STORE_DIR="./store/ca_only_diffusion_baseline_v2"
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

# ── Resolve checkpoint source ──────────────────────────────────────────────
# If --ckpt is passed, it overrides the config's baked-in ckpt_path/ckpt_name
# via Hydra CLI args. If not passed, we let the config's defaults take over.
CKPT_OVERRIDE=()
if [[ -n "$CKPT_PATH" ]]; then
    if [[ ! -f "$CKPT_PATH" ]]; then
        echo "ERROR: Checkpoint not found: $CKPT_PATH"
        exit 1
    fi
    CKPT_OVERRIDE=( "ckpt_path=$(dirname "$CKPT_PATH")" "ckpt_name=$(basename "$CKPT_PATH")" )
fi

echo "============================================"
echo "Config     : $CONFIG_NAME"
if [[ -n "$CKPT_PATH" ]]; then
    echo "Checkpoint : $CKPT_PATH (CLI override)"
else
    echo "Checkpoint : (using paths baked into $CONFIG_NAME.yaml)"
fi
echo "GPU        : $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "============================================"

# ── Clear previous inference results for this config ──────────────────────
INFER_DIR="./inference/${CONFIG_NAME}"
if [[ -d "$INFER_DIR" ]]; then
    echo "Removing old inference dir: $INFER_DIR"
    rm -rf "$INFER_DIR"
fi

# ── Generate ───────────────────────────────────────────────────────────────
# NOTE: generate.py and evaluate.py have DIFFERENT CLI conventions:
#   - generate.py uses @hydra.main → Hydra CLI: --config-name=X (hyphen),
#     plus `key=value` Hydra-style overrides.
#   - evaluate.py uses parse_args_and_cfg() → argparse: --config_name X
#     (underscore). Argparse is strict; ckpt_path=... overrides would
#     ERROR, so any ckpt override has to live in the YAML for evaluate.py.
echo ""
echo ">>> Generating samples..."
python proteinfoundation/generate.py \
    --config-name="$CONFIG_NAME" "${CKPT_OVERRIDE[@]}"

# ── Evaluate ───────────────────────────────────────────────────────────────
# evaluate.py reads ckpt_path/ckpt_name only from the config; CLI overrides
# are not honoured. The quick config has the right paths baked in.
echo ""
echo ">>> Evaluating samples (ESMFold designability, ~25 min on A100)..."
python proteinfoundation/evaluate.py \
    --config_name "$CONFIG_NAME"

echo ""
echo "Done. Results CSV: ./results_${CONFIG_NAME}_0.csv"
