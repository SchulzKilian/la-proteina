#!/bin/bash
#SBATCH -J ca_1gpu
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=6:00:00
#SBATCH --signal=SIGUSR1@300
#SBATCH --requeue
#SBATCH --output=slurm_ca_1gpu_%j.out

# CA-only training on a single Ampere A100 (80GB).
# Same env / RDS-fallback machinery as submit_train_1gpu.sh, but:
#   - default config is training_ca_only (no AE, no latents)
#   - no AE checkpoint staging (CA-only skips the AE via _ca_only_mode)
#   - lr is sqrt-scaled from the 4-GPU baseline (0.000415 -> 0.0002)
#   - accumulate_grad_batches=32 holds the 4-GPU effective batch on 1 GPU
#     (config batch_size=6, max_padding_size=512 → eff. batch ≈ 192)

# 1. Load personal shell config (env vars like WANDB_API_KEY).
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

# 3. Verify environment.
echo "Running on node: $(hostname)"
echo "Using Python: $(which python)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn) 2>/dev/null || true

# 3b. Pre-flight: RDS sometimes mounts read-only on individual compute nodes.
#     Instead of requeueing, fall back to /home for this job's checkpoint writes.
#     Sync back to RDS at job end if it becomes writable again.
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
    ln -sfn "$RDS_STORE" "$STORE_LINK" 2>/dev/null || true
}
trap cleanup_store EXIT

# 4. Override SLURM_NTASKS to match 1-GPU allocation.
#    full_training_test.sh defaults to SLURM_NTASKS=4; override it here.
export SLURM_NTASKS=1
export SLURM_NTASKS_PER_NODE=1

# 5. Skip srun — single-GPU doesn't need multi-task orchestration, and this
#    avoids inner-allocation hangs when running inside an interactive session.
#    full_training_test.sh uses `exec python ...` instead when NO_SRUN=1.
export NO_SRUN=1

# Auto-resume from the most recent run under ./store/<run_name>/. Requires a
# last.ckpt (or last-v<N>.ckpt) in that run's checkpoints/ dir — the resume
# logic in train.py:get_run_dirs keys off that filename, not best_val_*.
export RESUME=1

# 5b. wandb init needs more than 90s on Cambridge HPC compute nodes — the
#     outbound connection is flaky on first contact. Bump both init and
#     service-wait timeouts so training doesn't crash before it starts.
export WANDB_INIT_TIMEOUT=600
export WANDB__SERVICE_WAIT=300

# 6. Default config = training_ca_only. The user can still override with
#    `-n some_other_config` when invoking this script.
DEFAULT_CONFIG="training_ca_only"
has_n_flag=0
for arg in "$@"; do
    if [ "$arg" = "-n" ]; then has_n_flag=1; break; fi
done
if [ "$has_n_flag" -eq 0 ]; then
    set -- -n "$DEFAULT_CONFIG" "$@"
fi

# 6b. Group chained slots in wandb so the UI auto-aggregates them under one
#     row instead of showing N separate per-slot runs. Each chained slot
#     creates a new wandb run (run-ID isn't carried across slots), but if
#     they share WANDB_RUN_GROUP the dashboard treats them as one group with
#     a single curve when grouped, or stitches them on `trainer/global_step`
#     when not grouped. Group key = run_name_ from the YAML (falling back to
#     the config name), which is stable across slots of the same training.
CONFIG_NAME=""
prev=""
for arg in "$@"; do
    if [ "$prev" = "-n" ]; then CONFIG_NAME="$arg"; break; fi
    prev="$arg"
done
WANDB_GROUP="${CONFIG_NAME:-laproteina}"
if [ -n "$CONFIG_NAME" ] && [ -f "configs/${CONFIG_NAME}.yaml" ]; then
    yaml_run_name=$(awk -F': *' '/^run_name_:/ {print $2; exit}' "configs/${CONFIG_NAME}.yaml" \
                    | tr -d '"' | tr -d "'" | xargs)
    [ -n "$yaml_run_name" ] && WANDB_GROUP="$yaml_run_name"
fi
export WANDB_RUN_GROUP="$WANDB_GROUP"
echo "[+] wandb group: $WANDB_RUN_GROUP"

# 7. Run training with 1-GPU Hydra overrides tuned for CA-only:
#    - lr=0.0002 : constant LR, sqrt-scaled from 4-GPU baseline 0.000415.
#      No scheduler — the canonical recipe uses constant LR (the v2 attempt
#      with cosine_with_warmup is documented as a failure in CLAUDE.md).
#    - accumulate_grad_batches=32 : holds the 4-GPU effective batch
#      (4 * 8 * batch_size == 1 * 32 * batch_size).
#    - dist_strategy=auto : Lightning picks single-device, no DDP overhead.
#    - log.last_ckpt_every_n_steps=500 : more frequent last.ckpt writes
#      (~12-15 min apart on this setup). Caps worst-case progress loss if
#      SIGUSR1 save-on-preempt doesn't fire cleanly (auto_requeue is off
#      when RESUME is unset, so the Lightning SLURMEnvironment save path
#      is less reliable).
# Weight decay (0.05) lives in the YAML so the recipe is self-documenting
# and reproducible for the sparse-attention and conv-downsample variants
# that will reuse this config. DO NOT raise wd above 0.05 without first
# restructuring configure_optimizers to exclude AdaLN-Zero gates / LayerNorm
# / biases / embeddings from decay (see CLAUDE.md, Finding 6).
bash script_utils/full_training_test.sh "$@" \
    hardware.ngpus_per_node_=1 \
    hardware.nnodes_=1 \
    opt.dist_strategy=auto \
    opt.lr=0.0002 \
    opt.accumulate_grad_batches=32 \
    log.last_ckpt_every_n_steps=500
