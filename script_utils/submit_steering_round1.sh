#!/bin/bash
#SBATCH -J steer_r1_netchg
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=slurm_steer_r1_netchg_%j.out

# Do NOT use set -e on this cluster (SLURM TaskProlog mkdir fails)
set -uo pipefail

# Activate env (no source ~/.bashrc — BASHRCSOURCED unbound)
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env
# Point compute_developability to the TANGO binary (needed for aggregation metrics)
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release

cd /home/ks2218/la-proteina

echo "=== Round 1: net_charge maximize (real predictor, 5 samples) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

OUTPUT_DIR=results/steering_eval/round1_net_charge_up

# Step 1: Generate guided + unguided pairs
echo ""
echo "=== STEP 1: Generation (5 seeds × 1 length = 5 pairs = 10 samples) ==="
python -m steering.generate \
    --proteina_config inference_ucond_notri_long \
    --steering_config steering/config/net_charge_up_real.yaml \
    --n_samples 5 \
    --lengths 400 \
    --output_dir $OUTPUT_DIR \
    --device cuda:0 \
    --nsteps 100

# Step 2: Evaluate properties (CPU, skip TANGO since this round is just net_charge)
echo ""
echo "=== STEP 2: Property evaluation ==="
python -m steering.property_evaluate \
    --input_dir $OUTPUT_DIR

echo ""
echo "=== DONE ==="
ls -la $OUTPUT_DIR/ 2>&1

echo "=== unguided_properties.csv ==="
cat $OUTPUT_DIR/unguided_properties.csv 2>/dev/null
echo ""
echo "=== guided_properties.csv ==="
cat $OUTPUT_DIR/guided_properties.csv 2>/dev/null
echo ""
echo "=== report.txt ==="
cat $OUTPUT_DIR/report.txt 2>/dev/null
