#!/bin/bash
#SBATCH -J gen_eval_ca_only
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --output=slurm_eval_ca_%j.out

source $HOME/.bashrc
conda activate laproteina_env

cd $HOME/la-proteina

STORE_ROOT="./store"
PROJECT_NAME="ca_only_diffusion_70M"
CONFIG_NAME="inference_ucond_notri_ca_only"

# Optional: pass a full checkpoint path directly via -c /path/to/checkpoint.ckpt
EXPLICIT_CKPT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c) EXPLICIT_CKPT="$2"; shift 2 ;;
        *)  echo "Unknown argument: $1"; shift ;;
    esac
done

if [ -n "$EXPLICIT_CKPT" ]; then
    if [ ! -f "$EXPLICIT_CKPT" ]; then
        echo "Error: Checkpoint not found at $EXPLICIT_CKPT"
        exit 1
    fi
    CKPT_DIR=$(dirname "$EXPLICIT_CKPT")
    CKPT_NAME=$(basename "$EXPLICIT_CKPT")
    LATEST_JOB_ID="(manual)"
else
    # Find the latest run directory for this project
    PROJECT_DIR="${STORE_ROOT}/${PROJECT_NAME}"
    if [ ! -d "$PROJECT_DIR" ]; then
        echo "Error: Project directory $PROJECT_DIR not found."
        exit 1
    fi

    LATEST_JOB_ID=$(ls -t "$PROJECT_DIR" | head -1)
    CKPT_PATH="${PROJECT_DIR}/${LATEST_JOB_ID}"

    # Try last.ckpt (from EmaModelCheckpoint) first, fall back to hpc_ckpt_1.ckpt
    LAST_CKPT="${CKPT_PATH}/checkpoints/last.ckpt"
    HPC_CKPT="${CKPT_PATH}/hpc_ckpt_1.ckpt"

    if [ -f "$LAST_CKPT" ]; then
        CKPT_DIR="${CKPT_PATH}/checkpoints"
        CKPT_NAME="last.ckpt"
    elif [ -f "$HPC_CKPT" ]; then
        CKPT_DIR="${CKPT_PATH}"
        CKPT_NAME="hpc_ckpt_1.ckpt"
    else
        echo "Error: No checkpoint found. Tried:"
        echo "  $LAST_CKPT"
        echo "  $HPC_CKPT"
        exit 1
    fi
fi

echo "----------------------------------------------------------------"
echo "Project:    $PROJECT_NAME"
echo "Run:        $LATEST_JOB_ID"
echo "Checkpoint: ${CKPT_DIR}/${CKPT_NAME}"
echo "----------------------------------------------------------------"

# Clean previous inference run for this config (PDB output dir + results CSV)
rm -rf "inference/${CONFIG_NAME}"
rm -f "inference/results_${CONFIG_NAME}_"*.csv

# Generation
echo "Starting Generation..."
python proteinfoundation/generate.py \
    --config-name "$CONFIG_NAME" \
    ++config_name="$CONFIG_NAME" \
    ++ckpt_path="$CKPT_DIR" \
    ++ckpt_name="$CKPT_NAME"

# Evaluation
echo "Starting Evaluation..."
python proteinfoundation/evaluate.py --config_name "$CONFIG_NAME"

echo "Job Complete."
