#!/bin/bash
#SBATCH -J eval_baseline_L300_400
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_eval_baseline_L300_400_%j.out

source $HOME/.bashrc
# Cambridge HPC env lives at /home/ks2218/conda_envs/...; this L4 box has it
# at the standard ~/.conda/envs/... location. Probe both so the script is
# portable between the two environments.
if [[ -d /home/ks2218/conda_envs/laproteina_env ]]; then
    export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
elif [[ -d /home/ks2218/.conda/envs/laproteina_env ]]; then
    export LAPROTEINA_ENV=/home/ks2218/.conda/envs/laproteina_env
else
    echo "ERROR: laproteina_env not found in either location."
    exit 1
fi
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

# Long-length probe of the canonical baseline (step 2646, wd=0.05) — L=300, 400, N=3.
# Re-runs inference_2646_long (2026-04-25, pre-fix MPNN eval) with the post-fix
# eval path (designability.py ca_only=True, commit ed10dfe, 2026-04-28).
# nsteps=400 (inference_base default), seed=5, sc sampling — canonical recipe.

set -uo pipefail

CONFIG_NAME="inference_baseline_L300_400"
PMPNN_SCRIPT="./ProteinMPNN/protein_mpnn_run.py"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: No CUDA GPU available."
    exit 1
fi
if [[ ! -f "$PMPNN_SCRIPT" ]]; then
    echo "ERROR: ProteinMPNN weights not found at $PMPNN_SCRIPT"
    exit 1
fi

echo "============================================"
echo "Config : $CONFIG_NAME"
echo "Ckpt   : baseline_wd0.05_step2646.ckpt"
echo "Lengths: 300, 400 (N=3 each)"
echo "GPU    : $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "============================================"

INFER_DIR="./inference/${CONFIG_NAME}"
if [[ -d "$INFER_DIR" ]]; then
    echo "Removing old inference dir: $INFER_DIR"
    rm -rf "$INFER_DIR"
fi

# Defensive: stale CSV/FASTA will trip MMseqs2 "empty FASTA" if processed_latents was rebuilt.
rm -f "./inference/results_${CONFIG_NAME}_0.csv" \
      "./inference/df_pdb_${CONFIG_NAME}_latents.csv" \
      "./inference/seq_df_pdb_${CONFIG_NAME}_latents.fasta" 2>/dev/null

echo ""
echo ">>> Generating samples..."
# generate.py uses Hydra CLI (--config-name with hyphen);
# evaluate.py uses argparse (--config_name with underscore).
python proteinfoundation/generate.py --config-name="$CONFIG_NAME"

echo ""
echo ">>> Evaluating samples (ESMFold designability, post-fix MPNN ca_only=True)..."
python proteinfoundation/evaluate.py --config_name "$CONFIG_NAME"

echo ""
echo "Done. Results CSV: ./inference/results_${CONFIG_NAME}_0.csv"
