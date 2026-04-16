#!/bin/bash
#SBATCH -J train_1gpu
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --signal=SIGUSR1@300
#SBATCH --output=slurm_train_1gpu_%j.out

# 1. Load your personal shell config (Reliable Conda setup)
source $HOME/.bashrc

# 2. Activate the environment
conda activate laproteina_env

DATA_PATH="/home/ks2218/la-proteina/data"

# 3. Verify Environment
echo "Running on node: $(hostname)"
echo "Using Python: $(which python)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn) 2>/dev/null || true

# 4. Override SLURM_NTASKS to match 1-GPU allocation
#    full_training_test.sh hardcodes SLURM_NTASKS=4; override it here.
export SLURM_NTASKS=1
export SLURM_NTASKS_PER_NODE=1

# 5. Run training with 1-GPU Hydra overrides
#    lr=0.0005 is sqrt-scaled for 4x smaller effective batch vs 4-GPU default
#    (eff batch 832 → 208 => lr 0.001 → 0.0005)
bash script_utils/full_training_test.sh "$@" \
    hardware.ngpus_per_node_=1 \
    hardware.nnodes_=1 \
    opt.dist_strategy=auto \
    opt.lr=0.0005
