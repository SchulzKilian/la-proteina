#!/bin/bash
#SBATCH -J layer_sel
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_layer_sel_%j.out

# Layer-selective sparse-vs-dense inference diagnostic.
# Inference-only stress test on dense-trained weights (E019 step 2646 ckpt).
# Stage gating: halts if stage 1 controls don't reproduce baseline, or if
# stage 2 shows no lower-vs-upper asymmetry at K=128 L=50.
#
# Usage:
#   sbatch script_utils/submit_test_layer_selective_sparse_inference.sh         # all stages with halts
#   sbatch script_utils/submit_test_layer_selective_sparse_inference.sh 1       # stage 1 only
#   sbatch script_utils/submit_test_layer_selective_sparse_inference.sh 2       # stage 2 only (must have stage 1 in cells.json)

set -uo pipefail   # NOT -e: TaskProlog mkdir failures on this cluster kill -e scripts before runtime

source $HOME/.bashrc
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

# PYTHON_EXEC must be exported per [[feedback_export_python_exec]] — otherwise
# the ProteinMPNN subprocess inside designability eval gets bare `python` and
# scRMSD comes back uniformly -1.0.
export PYTHON_EXEC="${LAPROTEINA_ENV}/bin/python"

cd $HOME/la-proteina

echo "Running on $(hostname); GPU: $CUDA_VISIBLE_DEVICES; python: $(which python)"

STAGE_ARG="${1:-0}"   # 0 = all stages

# Use expandable_segments allocator — small win on gather-heavy bf16.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PYTHON_EXEC script_utils/test_layer_selective_sparse_inference.py --stage "$STAGE_ARG"
rc=$?

echo "Driver exit code: $rc"
exit $rc
