#!/bin/bash
# Local testing script (No SBATCH headers)

# 1. Setup environment
source $HOME/.bashrc
# Activate /home env via PATH prepend (NOT `conda activate`).
# /rds-based env hangs Python startup when any Lustre OST is evicted/disconn.
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env
export DATA_PATH=/home/ks2218/la-proteina/data
# 2. Run with Hydra overrides
# Using 'single=true' to trigger your existing local logic
python proteinfoundation/train.py \
    single=true \
    ++nolog=true \
    opt.max_epochs=1 \
    hardware.ncpus_per_task_train_=4 \
    "$@"