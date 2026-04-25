#!/bin/bash
#SBATCH -J gen_stratified
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:30:00
#SBATCH --output=slurm_gen_stratified_%j.out

# Length-stratified unguided generation: ~uniform coverage across 50-residue
# bins in [300, 800), n_per_bin=100 -> 1000 total samples.
# Replaces the empirical-distribution sampler in submit_generate_baseline.sh.
#
# Lengths are shuffled inside the script so that hitting the time limit
# costs proportionally from every bin, rather than dropping the upper bins.

set -uo pipefail

export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release

cd /home/ks2218/la-proteina

OUTPUT_DIR=${1:-results/generated_stratified_300_800}
N_PER_BIN=${2:-100}
NSTEPS=${3:-200}

echo "=== Length-stratified baseline generation ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "n_per_bin: $N_PER_BIN, bins: 10 (50-residue) over [300,800)"
echo "Total samples: $((N_PER_BIN * 10)), nsteps: $NSTEPS, output: $OUTPUT_DIR"

python -m steering.generate_baseline \
    --proteina_config inference_ucond_notri_long \
    --length_mode stratified \
    --n_per_bin $N_PER_BIN \
    --bin_width 50 \
    --length_range 300 800 \
    --output_dir $OUTPUT_DIR \
    --device cuda:0 \
    --nsteps $NSTEPS

rc=$?
echo "Generation finished (rc=$rc) at $(date)"
echo "=== Manifest head ==="
head -5 $OUTPUT_DIR/manifest.csv 2>/dev/null
echo "..."
echo "=== Count ==="
find $OUTPUT_DIR/samples -name '*.pt' 2>/dev/null | wc -l
echo ".pt files generated"

exit $rc
