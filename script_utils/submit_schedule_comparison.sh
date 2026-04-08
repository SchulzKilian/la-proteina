#!/bin/bash
#SBATCH -J schedule_cmp
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=slurm_schedule_cmp_%j.out

# submit_schedule_comparison.sh
#
# Runs generation + evaluation twice — baseline schedule vs proposed schedule —
# then prints a side-by-side designability comparison.
#
# Baseline : bb_ca log(p=2.0),   local_latents power(p=2.0)   [current defaults]
# Proposed : bb_ca power(p=0.5), local_latents power(p=0.5)   [curvature-aligned]
#
# Usage:
#   sbatch script_utils/submit_schedule_comparison.sh --config inference_ucond_ca_only_70M --ckpt /path/to/last.ckpt
#   bash   script_utils/submit_schedule_comparison.sh --config inference_ucond_notri       --ckpt /path/to/last.ckpt
#
# Flags:
#   --config  <name>   Hydra config name (without .yaml). Default: inference_ucond_ca_only_70M
#   --ckpt    <path>   Full path to checkpoint file. Required.
#   --ae_ckpt <path>   Full path to AE checkpoint (full-latent models only).

source "$HOME/.bashrc"
conda activate laproteina_env

set -euo pipefail

cd "$HOME/la-proteina"

# ── Defaults ──────────────────────────────────────────────────────────────────
CONFIG_NAME="inference_ucond_ca_only_70M"
CKPT_PATH=""
AE_CKPT_PATH=""
NSAMPLES=""

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)   CONFIG_NAME="$2";  shift 2 ;;
        --ckpt)     CKPT_PATH="$2";    shift 2 ;;
        --ae_ckpt)  AE_CKPT_PATH="$2"; shift 2 ;;
        --nsamples) NSAMPLES="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CKPT_PATH" ]]; then
    echo "ERROR: --ckpt <path> is required."
    exit 1
fi
if [[ ! -f "$CKPT_PATH" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

CKPT_DIR=$(dirname "$CKPT_PATH")
CKPT_NAME=$(basename "$CKPT_PATH")

# AE ckpt override string (empty for CA-only)
AE_OVERRIDE=""
if [[ -n "$AE_CKPT_PATH" ]]; then
    AE_OVERRIDE="++autoencoder_ckpt_path=$AE_CKPT_PATH"
fi

# nsamples override string (empty = use config default)
NSAMPLES_OVERRIDE=""
if [[ -n "$NSAMPLES" ]]; then
    NSAMPLES_OVERRIDE="++generation.dataset.nsamples=$NSAMPLES"
fi

# Where the two preserved result sets will live
CSV_BASELINE="inference/results_${CONFIG_NAME}_baseline_0.csv"
CSV_PROPOSED="inference/results_${CONFIG_NAME}_proposed_0.csv"

echo "================================================================"
echo "Schedule comparison"
echo "Config     : $CONFIG_NAME"
echo "Checkpoint : $CKPT_PATH"
echo "Baseline   : bb_ca log(p=2.0) | local_latents power(p=2.0)"
echo "Proposed   : bb_ca power(p=0.5) | local_latents power(p=0.5)"
echo "================================================================"

# ── Helper ────────────────────────────────────────────────────────────────────
# Both generate and evaluate must use the real CONFIG_NAME so that:
#   - generate.py writes PDBs to ./inference/CONFIG_NAME/
#   - evaluate.py loads configs/CONFIG_NAME.yaml and reads from ./inference/CONFIG_NAME/
# After each run we rename the PDB dir + results CSV to preserve them.
run_variant() {
    local label="$1"
    local save_csv="$2"   # where to save the CSV after evaluation
    local extra_overrides="$3"

    local infer_dir="inference/${CONFIG_NAME}"
    local csv_src="inference/results_${CONFIG_NAME}_0.csv"

    echo ""
    echo "── $label ─────────────────────────────────────────────────────"

    # Clear previous run outputs for this config_name
    rm -rf "$infer_dir"
    rm -f  "$csv_src"

    echo "Generating..."
    python proteinfoundation/generate.py \
        --config-name "$CONFIG_NAME" \
        ++config_name="$CONFIG_NAME" \
        ++ckpt_path="$CKPT_DIR" \
        ++ckpt_name="$CKPT_NAME" \
        $AE_OVERRIDE \
        $NSAMPLES_OVERRIDE \
        $extra_overrides

    echo "Evaluating..."
    python proteinfoundation/evaluate.py \
        --config_name "$CONFIG_NAME"

    # Preserve results CSV
    if [[ -f "$csv_src" ]]; then
        cp "$csv_src" "$save_csv"
        echo "Results saved to $save_csv"
    else
        echo "WARNING: expected CSV not found at $csv_src"
    fi
}

# ── Baseline (current schedule from config — no overrides needed) ─────────────
run_variant \
    "BASELINE  [bb_ca log p=2.0 | local_latents power p=2.0]" \
    "$CSV_BASELINE" \
    ""

# ── Proposed (curvature-aligned schedule) ────────────────────────────────────
run_variant \
    "PROPOSED  [bb_ca power p=0.5 | local_latents power p=0.5]" \
    "$CSV_PROPOSED" \
    "++generation.model.bb_ca.schedule.mode=power \
     ++generation.model.bb_ca.schedule.p=0.5 \
     ++generation.model.local_latents.schedule.mode=power \
     ++generation.model.local_latents.schedule.p=0.5"

# ── Summary comparison ────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "RESULTS COMPARISON"
echo "================================================================"
python - "$CSV_BASELINE" "$CSV_PROPOSED" <<'PYEOF'
import sys
import pandas as pd

csv_baseline, csv_proposed = sys.argv[1], sys.argv[2]

def load_results(path, label):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"  [!] No results CSV found: {path}")
        return None

KEY_METRICS = [
    "scRMSD_ca_esmfold_frac_below_2",
    "scRMSD_bb3o_esmfold_frac_below_2",
    "scTM_ca_esmfold_mean",
    "scTM_bb3o_esmfold_mean",
]

labels  = ["BASELINE", "PROPOSED"]
paths   = [csv_baseline, csv_proposed]
results = {}
for label, path in zip(labels, paths):
    df = load_results(path, label)
    if df is None:
        continue
    row = {}
    for m in KEY_METRICS:
        cols = [c for c in df.columns if m in c]
        if cols:
            row[m] = float(df[cols[0]].iloc[0])
    results[label] = row

if not results:
    print("No results found.")
    sys.exit(0)

all_metrics = sorted({m for r in results.values() for m in r})
col_w = 38
header = f"{'Metric':<{col_w}}" + "".join(f"{k:>12}" for k in labels) + "      delta"
print(header)
print("-" * len(header))
for m in all_metrics:
    row_str = f"{m:<{col_w}}"
    vals = [results.get(k, {}).get(m, float("nan")) for k in labels]
    for v in vals:
        row_str += f"{v:>12.4f}"
    if not any(v != v for v in vals):
        delta = vals[1] - vals[0]
        row_str += f"   {'+'if delta>=0 else ''}{delta:.4f}"
    print(row_str)
PYEOF

echo ""
echo "Raw CSVs:"
echo "  $CSV_BASELINE"
echo "  $CSV_PROPOSED"
echo "Done."
