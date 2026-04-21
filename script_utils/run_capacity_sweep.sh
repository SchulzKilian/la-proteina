#!/bin/bash
#SBATCH -J cap_probe_sweep
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:30:00
#SBATCH -o slurm_cap_probe_sweep_%j.out

# Do NOT use set -e (SLURM TaskProlog mkdir trips it).
set -uo pipefail

# Activate env (no source ~/.bashrc — triggers unbound variable with -u).
# Fallback order: HPC canonical path -> user-local ~/.conda path.
if [ -d /home/ks2218/conda_envs/laproteina_env ]; then
    export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
elif [ -d /home/ks2218/.conda/envs/laproteina_env ]; then
    export LAPROTEINA_ENV=/home/ks2218/.conda/envs/laproteina_env
else
    echo "ERROR: could not locate laproteina_env conda env" >&2
    exit 1
fi
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

echo "Node: $(hostname)"
echo "Python: $(which python)"
nvidia-smi | head -20

ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn) 2>/dev/null || true

cd /home/ks2218/la-proteina/laproteina_steerability

# Parse flags (passed through to the Python module).
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --folds) EXTRA_ARGS="$EXTRA_ARGS --folds $2"; shift 2 ;;
        --max-epochs) EXTRA_ARGS="$EXTRA_ARGS --max-epochs $2"; shift 2 ;;
        --batch-size) EXTRA_ARGS="$EXTRA_ARGS --batch-size $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "Extra args: $EXTRA_ARGS"
python -m src.capacity_probing.run_sweep --config config/default.yaml $EXTRA_ARGS
rc=$?

# Persist outputs to durable locations. /home is always available; RDS only on HPC.
if [ $rc -eq 0 ]; then
    RUN_DIR=$(ls -1dt logs/capacity_probing/*/ 2>/dev/null | head -1)
    if [ -n "$RUN_DIR" ]; then
        RUN_NAME=$(basename "$RUN_DIR")
        LOCAL_DST=/home/ks2218/la-proteina/checkpoints_capacity_probing/${RUN_NAME}
        mkdir -p "$LOCAL_DST"
        cp -r "${RUN_DIR}." "$LOCAL_DST/"
        ln -sfn "$LOCAL_DST" /home/ks2218/la-proteina/checkpoints_capacity_probing/latest
        echo "Persisted to: $LOCAL_DST"
        if [ -d /rds/user/ks2218/hpc-work ]; then
            RDS_DST=/rds/user/ks2218/hpc-work/checkpoints_capacity_probing/${RUN_NAME}
            mkdir -p "$RDS_DST"
            cp -r "${RUN_DIR}." "$RDS_DST/"
            ln -sfn "$RDS_DST" /rds/user/ks2218/hpc-work/checkpoints_capacity_probing/latest
            echo "Persisted to: $RDS_DST"
        fi
    fi
fi

exit $rc
