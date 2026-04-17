#!/bin/bash
#SBATCH -J train_1gpu
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
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

# 5. Skip srun — single-GPU doesn't need multi-task orchestration, and this
#    avoids inner-allocation hangs when running inside an interactive session.
#    full_training_test.sh uses `exec python ...` instead when NO_SRUN=1.
export NO_SRUN=1

# 5b. wandb init needs more than 90s on Cambridge HPC compute nodes — the
#     outbound connection is flaky on first contact. Bump both init and
#     service-wait timeouts so training doesn't crash before it starts.
export WANDB_INIT_TIMEOUT=600
export WANDB__SERVICE_WAIT=300

# 6. Stage AE checkpoint to local NVMe /tmp to avoid cold Lustre reads.
#    torch.load on a 4GB file from RDS is the biggest cold-start bottleneck.
#    One cp here (~30-60s) turns subsequent reads into fast local SSD reads.
AE_SRC="/rds/user/ks2218/hpc-work/checkpoints_laproteina/AE1_ucond_512.ckpt"
AE_LOCAL="/tmp/AE1_ucond_512.ckpt"
AE_OVERRIDE=""
if [ -f "$AE_SRC" ]; then
    echo "[+] Staging AE checkpoint to local /tmp..."
    t0=$(date +%s)
    cp "$AE_SRC" "$AE_LOCAL" && {
        t1=$(date +%s)
        echo "[+] Staged to $AE_LOCAL in $((t1 - t0))s"
        AE_OVERRIDE="++autoencoder_ckpt_path=$AE_LOCAL"
    } || {
        echo "[!] /tmp staging failed — falling back to RDS path"
    }
else
    echo "[!] Source AE ckpt not found at $AE_SRC — using config default"
fi

# 7. Run training with 1-GPU Hydra overrides
#    lr=0.0005 is sqrt-scaled for 4x smaller effective batch vs 4-GPU default
#    (eff batch 832 → 208 => lr 0.001 → 0.0005)
bash script_utils/full_training_test.sh "$@" \
    hardware.ngpus_per_node_=1 \
    hardware.nnodes_=1 \
    opt.dist_strategy=auto \
    opt.lr=0.0005 \
    $AE_OVERRIDE
