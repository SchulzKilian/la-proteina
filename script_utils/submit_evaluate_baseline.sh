#!/bin/bash
#SBATCH -J eval_baseline
#SBATCH -A COMPUTERLAB-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_eval_baseline_%j.out

# Do NOT use set -e (SLURM TaskProlog mkdir fails)
set -uo pipefail

export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release

cd /home/ks2218/la-proteina

# Args: positional, with defaults matching submit_generate_baseline.sh
SAMPLES_DIR=${1:-results/generated_baseline_300_800/samples}
OUTPUT_CSV=${2:-results/generated_baseline_300_800/properties_generated.csv}

echo "=== Developability evaluation on generated samples ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Samples dir: $SAMPLES_DIR"
echo "Output CSV: $OUTPUT_CSV"
echo "N .pt files: $(find $SAMPLES_DIR -name '*.pt' 2>/dev/null | wc -l)"

python -m steering.evaluate_samples_dir \
    --samples_dir $SAMPLES_DIR \
    --output_csv $OUTPUT_CSV
rc=$?

echo "Eval finished (rc=$rc) at $(date)"
echo "=== CSV head ==="
head -3 $OUTPUT_CSV 2>/dev/null
echo "=== Row count ==="
wc -l $OUTPUT_CSV

exit $rc
