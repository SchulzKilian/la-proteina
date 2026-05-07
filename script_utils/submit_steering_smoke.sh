#!/bin/bash
#SBATCH -J steering_smoke
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=slurm_steering_smoke_%j.out

# DO NOT use set -e (SLURM TaskProlog mkdir fails on this cluster)
set -uo pipefail

# Activate environment
source /rds/user/ks2218/hpc-work/conda_root/etc/profile.d/conda.sh
# Activate /home env via PATH prepend (NOT `conda activate`).
# /rds-based env hangs Python startup when any Lustre OST is evicted/disconn.
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release

cd /home/ks2218/la-proteina

echo "=== Steering smoke test ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

# Step 1: Generate guided + unguided proteins
echo ""
echo "=== STEP 1: Generation ==="
python -m steering.generate \
    --proteina_config inference_ucond_notri_long \
    --steering_config steering/config/smoke_test.yaml \
    --n_samples 1 \
    --lengths 300 \
    --output_dir results/steering_eval/smoke_test \
    --device cuda:0 \
    --nsteps 400  # was 100; flipped 2026-05-07 — nsteps<400 is below the integrator-convergence bar (22 Å vs 0.8 Å scRMSD cliff). See CLAUDE.md "Sampling — nsteps=400 is a HARD RULE".

# Step 2: Evaluate properties
echo ""
echo "=== STEP 2: Property evaluation ==="
python -m steering.property_evaluate \
    --input_dir results/steering_eval/smoke_test \
    --skip-tango

echo ""
echo "=== DONE ==="
echo "Results in results/steering_eval/smoke_test/"
cat results/steering_eval/smoke_test/report.txt 2>/dev/null
