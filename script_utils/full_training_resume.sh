#!/bin/bash
set -e 

# Setup Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -f .env ]; then source .env; fi
export SLURM_NTASKS=4

CHECKPOINT_PATH=$1
shift

echo "[+] Starting RESUME TRAINING from: $CHECKPOINT_PATH"
export SLURM_NTASKS_PER_NODE=$SLURM_NTASKS

# Launch training
srun python proteinfoundation/train.py \
    dataset=pdb/pdb_train_ucond \
    nn=local_latents_score_nn_160M \
    +pretrain_ckpt_path="$CHECKPOINT_PATH" \
    hydra.run.dir="logs/training_resume/$(date +%Y%m%d_%H%M%S)" \
    "$@"

echo "[+] Process Complete."