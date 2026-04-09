#!/usr/bin/env python3
"""
schedule_comparison_report.py  <baseline_csv> <proposed_csv>

Prints a side-by-side designability table and saves bar-chart PNGs comparing
two inference schedule runs produced by submit_schedule_comparison.sh.
"""
import sys
import math
import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_baseline, csv_proposed = sys.argv[1], sys.argv[2]

THRESHOLD = 2.0


def summarize(path, label):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"  [!] CSV not found: {path}")
        return None

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
        valid = scores[(scores > 0) & (scores < float("inf"))]
        result[f"{col}__n_valid"]      = len(valid)
        result[f"{col}__frac_below_2"] = float((valid < THRESHOLD).mean()) if len(valid) > 0 else float("nan")
        result[f"{col}__mean_rmsd"]    = float(valid.mean())               if len(valid) > 0 else float("nan")
    return result


# ── Load DataFrames ───────────────────────────────────────────────────────────
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
for label in dfs:
    path = csv_baseline if label == "BASELINE" else csv_proposed
    r = summarize(path, label)
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
        row_str += f"   {'+'if delta >= 0 else ''}{delta:.4f}"
    print(row_str)

# ── Plots ─────────────────────────────────────────────────────────────────────
rmsd_cols = sorted({
    c for df in dfs.values()
    for c in df.columns
    if c.startswith("_res_scRMSD_") and "_all_" not in c and not c.startswith("_res_co_")
})

if not rmsd_cols:
    print("No scRMSD columns found — skipping plots.")
    sys.exit(0)

for rmsd_col in rmsd_cols:
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

    all_lengths = sorted({int(l) for s in series.values() for l in s["L"].unique()})
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
            vals = s["scRMSD"].values if g == "Overall" else s.loc[s["L"] == int(g), "scRMSD"].values
            if len(vals) == 0:
                means_rmsd.append(float("nan")); stds_rmsd.append(0.0); means_frac.append(float("nan"))
            else:
                means_rmsd.append(float(np.mean(vals)))
                stds_rmsd.append(float(np.std(vals)))
                means_frac.append(float(np.mean(vals < 2.0)))

        xs = x + offsets[i]
        ax_rmsd.bar(xs, means_rmsd, bar_w * 0.9, yerr=stds_rmsd,
                    label=label, color=colors.get(label), capsize=4, alpha=0.85,
                    error_kw={"elinewidth": 1.2})
        ax_frac.bar(xs, means_frac, bar_w * 0.9, label=label, color=colors.get(label), alpha=0.85)

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
