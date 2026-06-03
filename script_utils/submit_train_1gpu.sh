#!/bin/bash
#SBATCH -J train_1gpu
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=8:00:00
#SBATCH --signal=SIGUSR1@300
#SBATCH --requeue
#SBATCH --output=slurm_train_1gpu_%j.out

# 1. Load your personal shell config (env vars like WANDB_API_KEY)
source $HOME/.bashrc

# 2. Activate the env via PATH prepend (NOT `conda activate`).
#    Why: the canonical env lives in /home/ks2218/conda_envs/laproteina_env,
#    not /rds. `conda activate` would re-resolve through conda's metadata which
#    in some setups still touches /rds — and any RDS Lustre OST eviction can
#    hang Python's stdlib import (fstat on .pyc files) for hours. Prepending
#    /home to PATH bypasses conda activation entirely.
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

DATA_PATH="/home/ks2218/la-proteina/data"

# 3. Verify Environment
echo "Running on node: $(hostname)"
echo "Using Python: $(which python)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn) 2>/dev/null || true

# 3b. Pre-flight: RDS sometimes mounts read-only on individual compute nodes.
#     Instead of requeueing, fall back to /home for this job's checkpoint
#     writes. Sync back to RDS at job end if it becomes writable again.
RDS_STORE="/rds/user/ks2218/hpc-work/store"
HOME_STORE="/home/ks2218/la-proteina/store_home_fallback"
STORE_LINK="/home/ks2218/la-proteina/store"
STORE_MODE="rds"

PROBE_FILE="$RDS_STORE/.write_probe_${SLURM_JOB_ID:-$$}_$(hostname)"
mkdir -p "$RDS_STORE" 2>/dev/null
if : > "$PROBE_FILE" 2>/dev/null; then
    rm -f "$PROBE_FILE"
    ln -sfn "$RDS_STORE" "$STORE_LINK"
    echo "[+] RDS writable on $(hostname) — using $RDS_STORE"
else
    STORE_MODE="home"
    echo "[!] RDS write probe FAILED on $(hostname) — falling back to /home"
    mkdir -p "$HOME_STORE"
    echo "[+] Staging last-*.ckpt from RDS -> /home (read should still work)..."
    t0=$(date +%s)
    # Only stage the last-* checkpoints (not chk_epoch_step_*). Saves ~5-15GB.
    rsync -a --prune-empty-dirs \
        --include='*/' --include='last.ckpt' --include='last-EMA.ckpt' \
        --include='last-v*.ckpt' --include='last-v*-EMA.ckpt' \
        --exclude='*' "$RDS_STORE/" "$HOME_STORE/" \
        && echo "[+] Stage done in $(($(date +%s) - t0))s" \
        || echo "[!] Staging failed — fresh run only, no resume possible"
    ln -sfn "$HOME_STORE" "$STORE_LINK"
    HOME_FREE=$(df -BG /home/ks2218 | awk 'NR==2{gsub(/G/,""); print $4}')
    echo "[!] Fallback active. /home free: ${HOME_FREE}G. Each ckpt pair ~5.6GB."
    echo "[!] After job: if RDS comes back, trap will auto-sync /home -> RDS."
fi

cleanup_store() {
    if [ "$STORE_MODE" = "home" ]; then
        # If RDS is writable again, sync new ckpts back so next job sees them.
        if touch "$RDS_STORE/.post_run_probe" 2>/dev/null; then
            rm -f "$RDS_STORE/.post_run_probe"
            echo "[cleanup] RDS writable again — syncing /home -> RDS..."
            rsync -a "$HOME_STORE/" "$RDS_STORE/" \
                && echo "[cleanup] Sync OK" \
                || echo "[cleanup] Sync failed — manual: rsync -a $HOME_STORE/ $RDS_STORE/"
        else
            echo "[cleanup] RDS still unwritable — ckpts remain in $HOME_STORE"
            echo "[cleanup] Sync manually when RDS recovers: rsync -a $HOME_STORE/ $RDS_STORE/"
        fi
    fi
    # Restore RDS symlink so the next job starts with the normal layout.
    ln -sfn "$RDS_STORE" "$STORE_LINK" 2>/dev/null || true
}
trap cleanup_store EXIT

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
#    lr=0.0003 + weight_decay=0.1 for 1-GPU: sqrt-scaled lr 0.0005 was overfitting;
#    dropping lr further and raising wd fights the smaller-effective-batch regime.
bash script_utils/full_training_test.sh "$@" \
    hardware.ngpus_per_node_=1 \
    hardware.nnodes_=1 \
    opt.dist_strategy=auto \
    opt.lr=0.0003 \
    +opt.weight_decay=0.1 \
    $AE_OVERRIDE
