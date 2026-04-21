#!/bin/bash
#SBATCH -J multitask_t1
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH -o slurm_multitask_t1_%j.out

# Do NOT use set -e on this cluster (SLURM TaskProlog mkdir fails)
set -uo pipefail

# NOTE: do NOT `source $HOME/.bashrc` — it references $BASHRCSOURCED which
# triggers `set -u` with unbound-variable. We set the env explicitly below.
# Activate /home env via PATH prepend (NOT `conda activate`).
# /rds-based env hangs Python startup when any Lustre OST is evicted/disconn.
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

echo "Running on node: $(hostname)"
echo "Using Python: $(which python)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"
nvidia-smi

ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn) 2>/dev/null || true

cd /home/ks2218/la-proteina/laproteina_steerability

# Parse arguments
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test) EXTRA_ARGS="$EXTRA_ARGS --smoke-test"; shift ;;
        --folds) EXTRA_ARGS="$EXTRA_ARGS --folds $2"; shift 2 ;;
        --batch-size) EXTRA_ARGS="$EXTRA_ARGS --batch-size $2"; shift 2 ;;
        --max-epochs) EXTRA_ARGS="$EXTRA_ARGS --max-epochs $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "Extra args: $EXTRA_ARGS"
python -m src.multitask_predictor.run --config config/default.yaml $EXTRA_ARGS
rc=$?

# Persist best checkpoint(s) to two durable locations (not just logs/).
# Find newest run directory; copy whole run (checkpoints + metrics CSVs + config).
if [ $rc -eq 0 ]; then
    RUN_DIR=$(ls -1dt logs/multitask_t1/*/ 2>/dev/null | head -1)
    if [ -n "$RUN_DIR" ] && [ -d "${RUN_DIR}checkpoints" ]; then
        RUN_NAME=$(basename "$RUN_DIR")
        LOCAL_DST=/home/ks2218/la-proteina/checkpoints_multitask_predictor/${RUN_NAME}
        RDS_DST=/rds/user/ks2218/hpc-work/checkpoints_multitask_predictor/${RUN_NAME}
        mkdir -p "$LOCAL_DST" "$RDS_DST"
        cp -r "${RUN_DIR}." "$LOCAL_DST/"
        cp -r "${RUN_DIR}." "$RDS_DST/"
        # Stable "latest" symlinks for easy programmatic access
        ln -sfn "$LOCAL_DST" /home/ks2218/la-proteina/checkpoints_multitask_predictor/latest
        ln -sfn "$RDS_DST" /rds/user/ks2218/hpc-work/checkpoints_multitask_predictor/latest
        echo "Persisted checkpoint to:"
        echo "  $LOCAL_DST"
        echo "  $RDS_DST"
    else
        echo "WARNING: could not locate run dir for checkpoint persistence (RUN_DIR=$RUN_DIR)"
    fi
fi

exit $rc
