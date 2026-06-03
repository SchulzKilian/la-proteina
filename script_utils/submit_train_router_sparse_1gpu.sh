#!/bin/bash
#SBATCH -J ca_router_K64
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=15:00:00
#SBATCH --signal=SIGUSR1@300
#SBATCH --requeue
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_ca_router_K64_%j.out

# Move-2 CA-only sparse trunk with frozen-router K=64 attention.
# Recipe matches submit_train_ca_only_1gpu.sh exactly except:
#   - default config is training_ca_only_router_sparse
#   - --exclude=gpu-q-43 added (CLAUDE.md: broken GPU, never schedule there)
#   - walltime 15h (single full slot per prompt; chain via --dependency)
#
# Source the user shell first, THEN any `set` commands. The E065 bug was the
# inverse order (`set -uo pipefail` before `source $HOME/.bashrc` → nounset
# triggered on unset BASHRCSOURCED inside .bashrc, killing the script before
# it executed). Canonical pattern below uses NO `set` flags at all — same as
# submit_train_ca_only_1gpu.sh.

# 1. Load personal shell config (env vars like WANDB_API_KEY).
source $HOME/.bashrc

# 2. Activate the env via PATH prepend (NOT `conda activate`).
#    See submit_train_ca_only_1gpu.sh for the /rds-Lustre rationale.
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

# 3b. RDS write-probe fallback (same pattern as canonical submit script).
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

# 4. SLURM_NTASKS / NO_SRUN (1-GPU mode).
export SLURM_NTASKS=1
export SLURM_NTASKS_PER_NODE=1
export NO_SRUN=1

# Auto-resume from the most recent run under ./store/<run_name>/.
export RESUME=1

# wandb timeouts (Cambridge HPC compute nodes are flaky on first wandb contact).
export WANDB_INIT_TIMEOUT=600
export WANDB__SERVICE_WAIT=300

# Allocator: expandable_segments helps gather-heavy bf16. Numerically a no-op.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Allow torch.compile to fire if cfg opts in via opt.compile_nn=True.
export TORCH_COMPILE_DISABLE=0

# 5. Default config = training_ca_only_router_sparse. Override with `-n some_cfg`.
DEFAULT_CONFIG="training_ca_only_router_sparse"
has_n_flag=0
for arg in "$@"; do
    if [ "$arg" = "-n" ]; then has_n_flag=1; break; fi
done
if [ "$has_n_flag" -eq 0 ]; then
    set -- -n "$DEFAULT_CONFIG" "$@"
fi

# 5b. Group chained slots in wandb by run_name_ (same pattern as canonical).
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

# 6. Hand off to full_training_test.sh with the canonical CA-only 1-GPU overrides.
#    Recipe values (lr=2e-4 constant, accumulate_grad_batches=32, wd in YAML)
#    match training_ca_only.yaml. No override on opt.weight_decay — the YAML
#    is the source of truth (so the recipe is self-documenting).
bash script_utils/full_training_test.sh "$@" \
    hardware.ngpus_per_node_=1 \
    hardware.nnodes_=1 \
    opt.dist_strategy=auto \
    opt.lr=0.0002 \
    opt.accumulate_grad_batches=32 \
    log.last_ckpt_every_n_steps=300
