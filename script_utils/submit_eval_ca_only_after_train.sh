#!/bin/bash
#SBATCH -J eval_ca_after_train
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_eval_ca_after_train_%j.out

# Post-training CA-only evaluation.
# Submit with:
#   sbatch --dependency=afterany:<train_jobid> script_utils/submit_eval_ca_only_after_train.sh
# afterany => runs on COMPLETED, FAILED, TIMEOUT, or CANCELLED (after start).
#
# CA-only correctness notes:
#   - inference_ucond_notri_ca_only config: autoencoder_ckpt_path is null (CA-only
#     asserts AE absent); generation sub-config uncond_codes_ca_only sets
#     designability_modes=[ca, bb3o] and compute_codesignability=False (poly-A
#     sequences from CA-only produce meaningless ESMFold refolds, so
#     codesignability / sequence-recovery are correctly disabled).
#   - ProteinMPNN CA-weights (ProteinMPNN/ca_model_weights/) are required for
#     ca-mode designability; verified present.
#   - Sizes [50,100,150,200,300,400,500] match the training length range
#     (min_length=50, max_length=512 in pdb_train_ucond.yaml).

set -o pipefail

# Source bashrc with -u off — /etc/bashrc line 12 references $BASHRCSOURCED
# before setting it, which trips `set -u` and aborts the job in ~6s.
set +u
source $HOME/.bashrc
set -u
# Activate /home env via PATH prepend (NOT `conda activate`).
# /rds-based env hangs Python startup when any Lustre OST is evicted/disconn.
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

cd $HOME/la-proteina || { echo "ERROR: cannot cd to $HOME/la-proteina"; exit 1; }

PROJECT_NAME="test_ca_only_diffusion"
CONFIG_NAME="inference_ucond_notri_ca_only"
STORE_DIR="./store/${PROJECT_NAME}"

# Training length range: min=50, max=512. These sizes span it.
# Override per-submission with --length and --nsamples.
LENGTHS="50,100,150,200,300,400,500"
NSAMPLES=20

# Run directory the training job is writing to. Pinned at SUBMIT time by
# the submitter (export EVAL_RUN_DIR=... before sbatch) so we always target the
# exact run the eval was queued against, even if a later training run starts
# writing a newer timestamped dir before this job fires.
RUN_DIR="${EVAL_RUN_DIR:-}"

# ── CLI overrides ──────────────────────────────────────────────────────────
# --ckpt <path/to/ckpt>     full checkpoint path (bypasses auto-pick below)
# --length "100" or "100,150"   comma-separated list for nres_lens
# --nsamples 20             samples per length
CLI_CKPT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)     CLI_CKPT="$2"; shift 2 ;;
        --length)   LENGTHS="$2"; shift 2 ;;
        --nsamples) NSAMPLES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "Running on node: $(hostname)"
echo "Using Python:    $(which python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# ── Resolve run directory ──────────────────────────────────────────────────
if [ -z "$RUN_DIR" ]; then
    echo "[!] EVAL_RUN_DIR not set — falling back to newest run under $STORE_DIR"
    if [ ! -d "$STORE_DIR" ]; then
        echo "ERROR: Project directory $STORE_DIR not found."
        exit 1
    fi
    RUN_DIR=$(ls -td "$STORE_DIR"/*/ 2>/dev/null | head -1)
    RUN_DIR="${RUN_DIR%/}"
fi

if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: Run directory not found: '$RUN_DIR'"
    exit 1
fi
# Resolve to absolute path so downstream config has no relative-path surprises.
RUN_DIR=$(readlink -f "$RUN_DIR")
CKPT_DIR="${RUN_DIR}/checkpoints"
if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CKPT_DIR"
    exit 1
fi

# ── Pick checkpoint (prefer non-EMA) ───────────────────────────────────────
# Preference order:
#   1. last.ckpt                    (non-EMA latest)
#   2. newest best_val_*.ckpt       (non-EMA, highest step)
#   3. last-EMA.ckpt                (EMA fallback)
#   4. newest best_val_*-EMA.ckpt   (EMA fallback)
# Skip files < 500MB — complete ckpt for a 158M-param model is ~1.8GB. A smaller
# file is likely a torch.save interrupted by TIMEOUT kill.
MIN_CKPT_SIZE=$((500 * 1024 * 1024))

is_valid_ckpt() {
    local f="$1"
    [ -f "$f" ] || return 1
    local sz
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$sz" -ge "$MIN_CKPT_SIZE" ]
}

pick_newest_valid() {
    local pattern="$1"
    local exclude="$2"   # regex to exclude (e.g. -EMA) or empty
    local candidates
    if [ -n "$exclude" ]; then
        candidates=$(ls -1 $pattern 2>/dev/null | grep -v -- "$exclude" | sort -V)
    else
        candidates=$(ls -1 $pattern 2>/dev/null | sort -V)
    fi
    # Iterate newest-first, return first valid
    for f in $(echo "$candidates" | tac); do
        if is_valid_ckpt "$f"; then
            echo "$f"
            return 0
        else
            echo "[!] Skipping undersized/corrupt checkpoint: $f ($(stat -c %s "$f" 2>/dev/null) bytes)" >&2
        fi
    done
    return 1
}

CKPT_NAME=""
if [ -n "$CLI_CKPT" ]; then
    # Explicit --ckpt wins.
    if [ ! -f "$CLI_CKPT" ]; then
        echo "ERROR: --ckpt path does not exist: $CLI_CKPT"; exit 1
    fi
    CKPT_DIR=$(dirname "$(readlink -f "$CLI_CKPT")")
    CKPT_NAME=$(basename "$CLI_CKPT")
else
    # Prefer newest best_val_*.ckpt. last.ckpt can be stale: Lightning's
    # save_last=True + a mid-epoch resume produces `last-v1.ckpt`, leaving
    # `last.ckpt` frozen at the pre-resume timestamp. By picking the newest
    # best_val first (which always reflects the most recent val-loss
    # improvement of the current session), we avoid evaluating a stale ckpt.
    BEST_RAW=$(pick_newest_valid "${CKPT_DIR}/best_val_*.ckpt" '-EMA') || true
    if [ -n "${BEST_RAW:-}" ]; then
        CKPT_NAME=$(basename "$BEST_RAW")
    elif is_valid_ckpt "${CKPT_DIR}/last.ckpt"; then
        CKPT_NAME="last.ckpt"
    elif is_valid_ckpt "${CKPT_DIR}/last-EMA.ckpt"; then
        CKPT_NAME="last-EMA.ckpt"
    else
        BEST_EMA=$(pick_newest_valid "${CKPT_DIR}/best_val_*-EMA.ckpt" "") || true
        if [ -n "${BEST_EMA:-}" ]; then
            CKPT_NAME=$(basename "$BEST_EMA")
        fi
    fi
fi

if [ -z "$CKPT_NAME" ]; then
    echo "ERROR: No checkpoint found in $CKPT_DIR."
    echo "       Training may have been cancelled before any checkpoint was written."
    exit 1
fi

CKPT_FULL="${CKPT_DIR}/${CKPT_NAME}"
if [ ! -f "$CKPT_FULL" ]; then
    echo "ERROR: Resolved checkpoint $CKPT_FULL does not exist."
    exit 1
fi

echo "----------------------------------------------------------------"
echo "Project     : $PROJECT_NAME"
echo "Run dir     : $RUN_DIR"
echo "Checkpoint  : $CKPT_FULL"
echo "Config      : $CONFIG_NAME"
echo "Lengths     : [$LENGTHS]"
echo "Nsamples/L  : $NSAMPLES"
echo "----------------------------------------------------------------"

# ── Generate a one-off Hydra config with overrides baked in ────────────────
# Why: proteinfoundation/generate.py uses argparse.parse_args() (strict) and
# hydra.compose(config_name=...) WITHOUT passing CLI overrides. So any `++foo=bar`
# or `foo=bar` trailing args would cause argparse to error out with
# "unrecognized arguments". The only reliable way to set ckpt_path / ckpt_name /
# nres_lens / nsamples is to bake them into a config file that inherits from
# the CA-only inference config.
GEN_CONFIG_NAME="inference_eval_ca_only_after_train_${SLURM_JOB_ID:-local}"
GEN_CONFIG_FILE="configs/${GEN_CONFIG_NAME}.yaml"

# Convert LENGTHS "50,100,150,..." to YAML list "[50, 100, 150, ...]"
YAML_LENGTHS="[$(echo "$LENGTHS" | sed 's/,/, /g')]"

cat > "$GEN_CONFIG_FILE" <<EOF
# Auto-generated by submit_eval_ca_only_after_train.sh
# DO NOT COMMIT — regenerated each job. Safe to delete.
defaults:
  - inference_ucond_notri_ca_only
  - _self_

run_name_: laproteina_eval_ca_after_train_${SLURM_JOB_ID:-local}
ckpt_path: ${CKPT_DIR}
ckpt_name: ${CKPT_NAME}
autoencoder_ckpt_path:

generation:
  dataset:
    nlens_cfg:
      nres_lens: ${YAML_LENGTHS}
    nsamples: ${NSAMPLES}
EOF

cleanup_config() {
    rm -f "$GEN_CONFIG_FILE"
}
trap cleanup_config EXIT

echo "Generated config: $GEN_CONFIG_FILE"
cat "$GEN_CONFIG_FILE"
echo "----------------------------------------------------------------"

# ── Clean prior inference outputs for this config ──────────────────────────
rm -rf "inference/${GEN_CONFIG_NAME}"
rm -f "inference/results_${GEN_CONFIG_NAME}_"*.csv
rm -f "results_${GEN_CONFIG_NAME}_"*.csv

# ── Generate ───────────────────────────────────────────────────────────────
echo ">>> Generating samples..."
python proteinfoundation/generate.py --config_name "$GEN_CONFIG_NAME"
GEN_RC=$?
if [ $GEN_RC -ne 0 ]; then
    echo "ERROR: Generation failed with exit code $GEN_RC"
    exit $GEN_RC
fi

# ── Evaluate (ProteinMPNN-CA + ESMFold on bb3o/ca designability) ───────────
echo ">>> Evaluating samples..."
python proteinfoundation/evaluate.py --config_name "$GEN_CONFIG_NAME"
EVAL_RC=$?
if [ $EVAL_RC -ne 0 ]; then
    echo "ERROR: Evaluation failed with exit code $EVAL_RC"
    exit $EVAL_RC
fi

echo "Job complete. Results: results_${GEN_CONFIG_NAME}_0.csv"
