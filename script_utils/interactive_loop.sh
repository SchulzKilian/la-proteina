#!/bin/bash
# Run this inside a tmux session on the login node:
#   tmux new -s training
#   bash script_utils/interactive_loop.sh -n training_ca_only_70M
#
# It loops: request GPU allocation → train → save checkpoint → release → repeat.
# Ctrl+C to stop the loop (current session finishes first).

# NO set -e here — salloc exits non-zero when job hits time limit, which is normal.

CONFIG_NAME="training_ca_only_70M"
TIME="1:00:00"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) CONFIG_NAME="$2"; shift 2 ;;
    -t) TIME="$2"; shift 2 ;;
    *) shift ;;
  esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "[loop] Config: $CONFIG_NAME  Time: $TIME"
echo "[loop] Starting at $(date). Ctrl+C to stop after current session."

while true; do
    echo ""
    echo "[loop] Requesting allocation at $(date)..."

    # --signal=SIGUSR1@300: SLURM sends SIGUSR1 5 min before time limit.
    # srun forwards it to the Python tasks → Lightning saves last.ckpt and exits.
    # salloc will exit non-zero after SIGUSR1 — that is expected and fine.
    salloc \
        --account=COMPUTERLAB-SL3-GPU \
        -p ampere \
        --gres=gpu:4 \
        --nodes=1 \
        --ntasks-per-node=4 \
        --cpus-per-task=16 \
        --mem=500G \
        --time="$TIME" \
        --signal=SIGUSR1@300 \
        bash -c "
            source \$HOME/.bashrc
            # Activate /home env via PATH prepend (NOT `conda activate`).
# /rds-based env hangs Python startup when any Lustre OST is evicted/disconn.
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env
            cd '$PROJECT_DIR'
            bash script_utils/full_training_test.sh -n '$CONFIG_NAME'
        " || true  # swallow non-zero exit so the loop continues

    echo "[loop] Session ended at $(date). Waiting 30s before next allocation..."
    sleep 30
done
