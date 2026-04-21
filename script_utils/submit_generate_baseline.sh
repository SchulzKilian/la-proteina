#!/bin/bash
#SBATCH -J gen_baseline
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=slurm_gen_baseline_%j.out

# Do NOT use set -e (SLURM TaskProlog mkdir fails)
set -uo pipefail

export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release

cd /home/ks2218/la-proteina

# Default args — override via sbatch command line if needed
N_SAMPLES=${1:-100}
OUTPUT_DIR=${2:-results/generated_baseline_300_800}
NSTEPS=${3:-200}

echo "=== Unguided baseline generation (length-matched 300-800) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "N samples: $N_SAMPLES, nsteps: $NSTEPS, output: $OUTPUT_DIR"

python -m steering.generate_baseline \
    --proteina_config inference_ucond_notri_long \
    --n_samples $N_SAMPLES \
    --length_csv /rds/user/ks2218/hpc-work/developability_panel.csv \
    --length_col sequence_length \
    --length_range 300 800 \
    --output_dir $OUTPUT_DIR \
    --device cuda:0 \
    --nsteps $NSTEPS

rc=$?
echo "Generation finished (rc=$rc) at $(date)"

# Quick summary
echo "=== Manifest head ==="
head -5 $OUTPUT_DIR/manifest.csv 2>/dev/null
echo "..."
echo "=== Count ==="
find $OUTPUT_DIR/samples -name '*.pt' 2>/dev/null | wc -l
echo ".pt files generated"

exit $rc
