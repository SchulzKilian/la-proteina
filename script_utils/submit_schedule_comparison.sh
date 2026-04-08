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
import math
import pandas as pd

csv_baseline, csv_proposed = sys.argv[1], sys.argv[2]

THRESHOLD = 2.0

def summarize(path, label):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"  [!] CSV not found: {path}")
        return None

    # Auto-detect all scRMSD designability columns (e.g. _res_scRMSD_ca_esmfold)
    # Exclude codesignability cols (_res_co_scRMSD_*) and per-sample-list cols (_res_scRMSD_all_*)
    rmsd_cols = [
        c for c in df.columns
        if c.startswith("_res_scRMSD_")
        and "_all_" not in c
        and not c.startswith("_res_co_")
    ]
    if not rmsd_cols:
        print(f"  [!] No scRMSD columns found in {path}.")
        print(f"      Available _res* cols: {[c for c in df.columns if c.startswith('_res')]}")
        return None

    result = {"n_total": len(df)}
    for col in sorted(rmsd_cols):
        scores = pd.to_numeric(df[col], errors="coerce")
        # Match evaluate.py filter: score > 0 and score < inf
        valid = scores[(scores > 0) & (scores < float("inf"))]
        result[f"{col}__n_valid"]       = len(valid)
        result[f"{col}__frac_below_2"]  = float((valid < THRESHOLD).mean()) if len(valid) > 0 else float("nan")
        result[f"{col}__mean_rmsd"]     = float(valid.mean())               if len(valid) > 0 else float("nan")
    return result

# Load raw DataFrames (needed for per-length breakdown and plots)
dfs = {}
for label, path in [("BASELINE", csv_baseline), ("PROPOSED", csv_proposed)]:
    try:
        dfs[label] = pd.read_csv(path)
    except FileNotFoundError:
        print(f"  [!] CSV not found: {path}")

if not dfs:
    print("No results found.")
    sys.exit(0)

results = {}
for label, df in dfs.items():
    r = summarize(csv_baseline if label == "BASELINE" else csv_proposed, label)
    if r:
        results[label] = r

# ── Text table ────────────────────────────────────────────────────────────────
all_keys = []
seen = set()
for r in results.values():
    for k in r:
        if k not in seen:
            all_keys.append(k)
            seen.add(k)

labels_ordered = ["BASELINE", "PROPOSED"]
col_w = 42
header = f"{'Metric':<{col_w}}" + "".join(f"{k:>12}" for k in labels_ordered if k in results) + "      delta"
print(header)
print("-" * len(header))

for key in all_keys:
    row_str = f"{key:<{col_w}}"
    vals = [results.get(lab, {}).get(key, float("nan")) for lab in labels_ordered if lab in results]
    for v in vals:
        if isinstance(v, int):
            row_str += f"{v:>12d}"
        elif math.isnan(v):
            row_str += f"{'nan':>12}"
        else:
            row_str += f"{v:>12.4f}"
    if len(vals) == 2 and not any(math.isnan(v) for v in vals) and not isinstance(vals[0], int):
        delta = vals[1] - vals[0]
        row_str += f"   {'+'if delta>=0 else ''}{delta:.4f}"
    print(row_str)

# ── Plots ─────────────────────────────────────────────────────────────────────
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Find the scRMSD columns present in either dataframe
rmsd_cols = sorted({
    c for df in dfs.values()
    for c in df.columns
    if c.startswith("_res_scRMSD_") and "_all_" not in c and not c.startswith("_res_co_")
})

if not rmsd_cols:
    print("No scRMSD columns found — skipping plots.")
    sys.exit(0)

for rmsd_col in rmsd_cols:
    # Collect per-sample (length, scRMSD) for each label
    series = {}
    for label, df in dfs.items():
        if rmsd_col not in df.columns or "L" not in df.columns:
            continue
        scores = pd.to_numeric(df[rmsd_col], errors="coerce")
        valid_mask = (scores > 0) & (scores < float("inf"))
        series[label] = df.loc[valid_mask, ["L"]].copy()
        series[label]["scRMSD"] = scores[valid_mask].values

    if not series:
        continue

    # Groups: each unique length + "Overall"
    all_lengths = sorted({
        int(l) for s in series.values() for l in s["L"].unique()
    })
    groups = [str(l) for l in all_lengths] + ["Overall"]

    colors = {"BASELINE": "#4e79a7", "PROPOSED": "#e05c5c"}
    n_labels = len(series)
    bar_w = 0.35
    offsets = np.linspace(-(n_labels - 1) * bar_w / 2, (n_labels - 1) * bar_w / 2, n_labels)
    x = np.arange(len(groups))

    fig, (ax_rmsd, ax_frac) = plt.subplots(1, 2, figsize=(max(10, 3 * len(groups)), 5),
                                            constrained_layout=True)

    for i, (label, s) in enumerate(series.items()):
        means_rmsd, stds_rmsd, means_frac = [], [], []
        for g in groups:
            if g == "Overall":
                vals = s["scRMSD"].values
            else:
                vals = s.loc[s["L"] == int(g), "scRMSD"].values
            if len(vals) == 0:
                means_rmsd.append(float("nan"))
                stds_rmsd.append(0.0)
                means_frac.append(float("nan"))
            else:
                means_rmsd.append(float(np.mean(vals)))
                stds_rmsd.append(float(np.std(vals)))
                means_frac.append(float(np.mean(vals < 2.0)))

        xs = x + offsets[i]
        ax_rmsd.bar(xs, means_rmsd, bar_w * 0.9, yerr=stds_rmsd,
                    label=label, color=colors.get(label, None),
                    capsize=4, alpha=0.85, error_kw={"elinewidth": 1.2})
        ax_frac.bar(xs, means_frac, bar_w * 0.9,
                    label=label, color=colors.get(label, None), alpha=0.85)

    col_short = rmsd_col.replace("_res_scRMSD_", "").replace("_", " ")

    ax_rmsd.axhline(2.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="2Å threshold")
    ax_rmsd.set_xticks(x); ax_rmsd.set_xticklabels(groups)
    ax_rmsd.set_xlabel("Protein length (residues)")
    ax_rmsd.set_ylabel("mean scRMSD (Å) ± std")
    ax_rmsd.set_title(f"scRMSD — {col_short}")
    ax_rmsd.legend(fontsize=8)

    ax_frac.set_xticks(x); ax_frac.set_xticklabels(groups)
    ax_frac.set_xlabel("Protein length (residues)")
    ax_frac.set_ylabel("Fraction designable (scRMSD < 2Å)")
    ax_frac.set_title(f"Designability — {col_short}")
    ax_frac.set_ylim(0, 1.05)
    ax_frac.legend(fontsize=8)

    out_png = os.path.join(
        os.path.dirname(csv_baseline),
        f"schedule_comparison_{rmsd_col.replace('_res_scRMSD_', '')}.png"
    )
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out_png}")
PYEOF

echo ""
echo "Raw CSVs:"
echo "  $CSV_BASELINE"
echo "  $CSV_PROPOSED"
echo "Done."
