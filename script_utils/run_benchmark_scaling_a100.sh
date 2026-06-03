#!/bin/bash
# Inference compute-scaling benchmark on A100 (interactive session).
# Extends E074 (L4, L=50..500) to long lengths + A100, both arms, L=100 until
# dense OOMs. Run this INSIDE an interactive GPU session, e.g.:
#
#   srun -A COMPUTERLAB-SL3-GPU -p ampere --gres=gpu:1 --cpus-per-task=16 \
#        --mem=64G -t 8:00:00 --exclude=gpu-q-43 --pty bash
#   bash script_utils/run_benchmark_scaling_a100.sh
#
# Env-overridable: LENGTHS, NSAMPLES, SEED, OUTPUT_CSV, CUDA_VISIBLE_DEVICES, ARMS.
#
# Cost warning: dense wall ~ L^1.7 and runs the full nsteps=400 ODE at each
# length it fits. Near the A100-80GB ceiling (extrapolated L~2200) a single
# dense length is several minutes; the high-L tail can total 1-2 h. The CSV is
# written row-by-row, so Ctrl-C / time-limit keeps every length measured so far.
# Sparse runs its full ladder regardless of where dense stops.

set -uo pipefail

cd "$(dirname "$0")/.."  # repo root

# --- env (PATH prepend, NOT `conda activate`; HPC env lives in /home) ---
if [[ -d /home/ks2218/conda_envs/laproteina_env ]]; then
    export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
elif [[ -d /home/ks2218/.conda/envs/laproteina_env ]]; then
    export LAPROTEINA_ENV=/home/ks2218/.conda/envs/laproteina_env
else
    echo "ERROR: laproteina_env not found in either location." >&2
    exit 1
fi
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

# Helps gather-heavy bf16 (sparse path) per CLAUDE.md.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# --- knobs ---
LENGTHS=${LENGTHS:-"100 200 300 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2200 2400"}
NSAMPLES=${NSAMPLES:-2}
SEED=${SEED:-5}
ARMS=${ARMS:-"canonical_dense sparse_K40"}
OUTPUT_CSV=${OUTPUT_CSV:-results/inference_compute_audit/scaling_a100.csv}

# --- guards ---
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: No CUDA GPU visible. Are you in an interactive GPU session?" >&2
    exit 1
fi
# Arm-aware ckpt guard. CA-only arms expect repo-root symlinks (ckpt_path=repo
# root); official_LD3 ships its own ckpt_path=./checkpoints_laproteina, so it is
# checked there instead.
for arm in $ARMS; do
    case "$arm" in
        canonical_dense) need="baseline_wd0.05_step2646.ckpt" ;;
        sparse_K40)      need="sparse_K40_step1259.ckpt" ;;
        official_LD3)    need="checkpoints_laproteina/LD3_ucond_notri_800.ckpt
checkpoints_laproteina/AE2_ucond_800.ckpt" ;;
        *) echo "ERROR: unknown arm '$arm'" >&2; exit 1 ;;
    esac
    for ckpt in $need; do
        if [[ ! -f "$ckpt" ]]; then
            echo "ERROR: arm '$arm' needs checkpoint '$ckpt' (not found from $(pwd))." >&2
            exit 1
        fi
    done
done

echo "[run] GPU=$(python -c 'import torch;print(torch.cuda.get_device_name(0))')"
echo "[run] lengths=$LENGTHS"
echo "[run] arms=$ARMS  nsamples=$NSAMPLES  seed=$SEED"
echo "[run] output=$OUTPUT_CSV"

python script_utils/benchmark_inference_scaling.py \
    --output_csv "$OUTPUT_CSV" \
    --nsamples "$NSAMPLES" \
    --seed "$SEED" \
    --arms $ARMS \
    --lengths $LENGTHS

echo "[run] done. CSV: $OUTPUT_CSV"
