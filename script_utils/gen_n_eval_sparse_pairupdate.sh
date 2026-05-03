#!/bin/bash
#SBATCH -J eval_sparse_pairupdate
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_eval_sparse_pairupdate_%j.out

source $HOME/.bashrc
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

# Quick designability probe for the sparse K=40 + pair-update CA-only ckpt
# (ca_only_sparse_K40_pairupdate/1777463843, best_val ep=11 step=1133).
# 3 lengths × 6 samples × 200 ODE steps; ~15 min on 1× L4 (cf. E018 paramgroups N=6 ≈ 16 min).

set -uo pipefail

CONFIG_NAME="inference_sparse_pairupdate_quick"
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
python proteinfoundation/generate.py --config_name="$CONFIG_NAME"

echo ""
echo ">>> Evaluating samples (ESMFold designability)..."
python proteinfoundation/evaluate.py --config_name "$CONFIG_NAME"

echo ""
echo "Done. Results CSV: ./results_${CONFIG_NAME}_0.csv"
