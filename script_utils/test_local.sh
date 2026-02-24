#!/bin/bash
# Local testing script (No SBATCH headers)

# 1. Setup environment
source $HOME/.bashrc
conda activate laproteina_env

# 2. Run with Hydra overrides
# Using 'single=true' to trigger your existing local logic
python proteinfoundation/train.py \
    single=true \
    +nolog=true \
    opt.max_epochs=1 \
    hardware.ncpus_per_task_train_=4 \
    "$@"