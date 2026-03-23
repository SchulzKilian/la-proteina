#!/bin/bash
#SBATCH -J gen_eval_ca_only
#SBATCH -A COMPUTERLAB-SL2-GPU
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
PROJECT_NAME="test_ca_only_diffusion"
CONFIG_NAME="inference_ucond_notri"

# Find latest run for this project
PROJECT_DIR="${STORE_ROOT}/${PROJECT_NAME}"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project directory $PROJECT_DIR not found."
    exit 1
fi

LATEST_JOB_ID=$(ls -t "$PROJECT_DIR" | head -1)
CKPT_PATH="${PROJECT_DIR}/${LATEST_JOB_ID}/checkpoints"
CKPT_FILE="${CKPT_PATH}/last.ckpt"

if [ ! -f "$CKPT_FILE" ]; then
    echo "Error: Checkpoint not found at $CKPT_FILE"
    exit 1
fi

echo "----------------------------------------------------------------"
echo "Project:    $PROJECT_NAME"
echo "Run:        $LATEST_JOB_ID"
echo "Checkpoint: $CKPT_FILE"
echo "----------------------------------------------------------------"

# Clean previous inference runs
rm -rf inference

# Generation
echo "Starting Generation..."
python proteinfoundation/generate.py \
    --config-name "$CONFIG_NAME" \
    ++config_name="$CONFIG_NAME" \
    ++ckpt_path="$CKPT_PATH" \
    ++ckpt_name="last.ckpt"

# Evaluation
echo "Starting Evaluation..."
python proteinfoundation/evaluate.py --config_name "$CONFIG_NAME"

echo "Job Complete."
