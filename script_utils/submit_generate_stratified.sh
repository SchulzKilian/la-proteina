#!/bin/bash
#SBATCH -J gen_stratified
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=11:59:00
#SBATCH --output=slurm_gen_stratified_%j.out

# Length-stratified unguided generation: ~uniform coverage across 50-residue
# bins in [300, 800). Runs until the SLURM walltime is about to expire and
# appends to --output_dir, so re-submitting the same job keeps growing the
# sample population. Bins are filled round-robin in steering/generate_baseline.py
# so an early kill leaves all bins equally populated.

set -uo pipefail

export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release

cd /home/ks2218/la-proteina

OUTPUT_DIR=${1:-results/generated_stratified_300_800}
NSTEPS=${2:-200}

echo "=== Length-stratified baseline generation (run-until-timeout) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "nsteps: $NSTEPS, output: $OUTPUT_DIR"
echo "SLURM_JOB_END_TIME=${SLURM_JOB_END_TIME:-unset}"

python -m steering.generate_baseline \
    --proteina_config inference_ucond_notri_long \
    --length_mode stratified \
    --bin_width 50 \
    --length_range 300 800 \
    --output_dir $OUTPUT_DIR \
    --device cuda:0 \
    --nsteps $NSTEPS \
    --run_until_timeout \
    --slurm_safety_s 180

rc=$?
echo "Generation finished (rc=$rc) at $(date)"
echo "=== Manifest head ==="
head -5 $OUTPUT_DIR/manifest.csv 2>/dev/null
echo "..."
echo "=== Count ==="
find $OUTPUT_DIR/samples -name '*.pt' 2>/dev/null | wc -l
echo ".pt files generated"

exit $rc
