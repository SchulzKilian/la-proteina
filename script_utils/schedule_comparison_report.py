#!/usr/bin/env python3
"""
schedule_comparison_report.py  "LABEL1:path1.csv" "LABEL2:path2.csv" ...

Prints a side-by-side designability table and saves bar-chart PNGs comparing
N inference schedule runs produced by submit_schedule_comparison.sh.

Each argument is "LABEL:path" (colon-separated).  The first entry is treated
as the baseline for delta calculations.
"""
import sys
import math
import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binomtest

# ── Parse arguments ───────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: schedule_comparison_report.py 'LABEL1:path1' 'LABEL2:path2' ...")
    sys.exit(1)

entries = []
for arg in sys.argv[1:]:
    if ":" in arg:
        label, path = arg.split(":", 1)
    else:
        label = os.path.splitext(os.path.basename(arg))[0]
        path = arg
    entries.append((label.strip(), path.strip()))

THRESHOLD = 2.0

PALETTE = ["#4e79a7", "#e05c5c", "#59a14f", "#f28e2b", "#b07aa1", "#9c755f", "#76b7b2"]


def summarize(path, label):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"  [!] CSV not found: {path}")
        return None

    rmsd_cols = [
        c for c in df.columns
        if c.startswith("_res_scRMSD_") or c.startswith("_res_co_scRMSD_")
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
results = {}
labels_ordered = []

for label, path in entries:
    try:
        df = pd.read_csv(path)
        dfs[label] = df
        labels_ordered.append(label)
    except FileNotFoundError:
        print(f"  [!] CSV not found: {path}")

    r = summarize(path, label)
    if r:
        results[label] = r

if not dfs:
    print("No results found.")
    sys.exit(0)

# ── Text table ────────────────────────────────────────────────────────────────
all_keys = []
seen = set()
for r in results.values():
    for k in r:
        if k not in seen:
            all_keys.append(k)
            seen.add(k)

present = [l for l in labels_ordered if l in results]
baseline_label = present[0] if present else None

col_w = 42
val_w = 14
header = f"{'Metric':<{col_w}}" + "".join(f"{l[:val_w]:>{val_w}}" for l in present)
if len(present) >= 2:
    header += "".join(f"{'Δ vs baseline':>{val_w}}" for _ in present[1:])
print(header)
print("-" * len(header))

for key in all_keys:
    row_str = f"{key:<{col_w}}"
    vals = [results.get(lab, {}).get(key, float("nan")) for lab in present]
    for v in vals:
        if isinstance(v, int):
            row_str += f"{v:>{val_w}d}"
        elif isinstance(v, float) and math.isnan(v):
            row_str += f"{'nan':>{val_w}}"
        else:
            row_str += f"{v:>{val_w}.4f}"
    # Delta columns (each proposed vs baseline)
    if len(vals) >= 2 and not isinstance(vals[0], int):
        base_val = vals[0]
        for v in vals[1:]:
            if not math.isnan(v) and not math.isnan(base_val):
                delta = v - base_val
                row_str += f"   {'+'if delta >= 0 else ''}{delta:.4f}"
            else:
                row_str += f"{'nan':>{val_w}}"
    print(row_str)

# ── Paired analysis (id_gen pairs same initial noise across schedules) ───────
def _paired_scrmsd_stats(base, variant):
    """Per-pair Δ = variant - base on the SAME id_gen. Negative Δ = variant better.
    Returns dict of summary stats; nan-fields if insufficient data."""
    valid = (base > 0) & (base < np.inf) & (variant > 0) & (variant < np.inf)
    base_v, var_v = base[valid].to_numpy(), variant[valid].to_numpy()
    n = len(base_v)
    out = {"n_pairs": n}
    if n == 0:
        return {**out, "mean_delta": np.nan, "median_delta": np.nan,
                "frac_variant_better": np.nan, "wilcoxon_p": np.nan, "sign_p": np.nan}
    delta = var_v - base_v
    n_better = int((delta < 0).sum())
    n_nonzero = int((delta != 0).sum())
    try:
        w_p = float(wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue) if n_nonzero >= 1 else np.nan
    except ValueError:
        w_p = np.nan
    sign_p = float(binomtest(n_better, n_nonzero, p=0.5, alternative="two-sided").pvalue) if n_nonzero >= 1 else np.nan
    return {**out, "mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)),
            "frac_variant_better": n_better / n, "wilcoxon_p": w_p, "sign_p": sign_p}


def _paired_designability_stats(base, variant, threshold=THRESHOLD):
    """McNemar (exact binomial on discordant pairs) on binary designability.
    Returns dict; positive Δ rate = variant designable more often than base."""
    valid = (base > 0) & (base < np.inf) & (variant > 0) & (variant < np.inf)
    b = (base[valid] < threshold).to_numpy()
    v = (variant[valid] < threshold).to_numpy()
    n = len(b)
    out = {"n_pairs": n}
    if n == 0:
        return {**out, "base_designable": np.nan, "variant_designable": np.nan,
                "discordant_v_only": 0, "discordant_b_only": 0, "mcnemar_p": np.nan}
    v_only = int(((~b) & v).sum())   # variant rescues a baseline failure
    b_only = int((b & (~v)).sum())   # variant breaks a baseline success
    discordant = v_only + b_only
    mcnemar_p = float(binomtest(v_only, discordant, p=0.5, alternative="two-sided").pvalue) if discordant >= 1 else np.nan
    return {**out, "base_designable": float(b.mean()), "variant_designable": float(v.mean()),
            "discordant_v_only": v_only, "discordant_b_only": b_only, "mcnemar_p": mcnemar_p}


paired_rmsd_cols = sorted({
    c for df in dfs.values()
    for c in df.columns
    if c.startswith("_res_scRMSD_") or c.startswith("_res_co_scRMSD_")
})

if len(present) >= 2 and paired_rmsd_cols and all("id_gen" in dfs[l].columns for l in present):
    paired_rows = []
    base_df = dfs[baseline_label].set_index("id_gen")
    print()
    print("=" * 80)
    print("PAIRED ANALYSIS  (same id_gen ⇒ same initial noise; Δ = variant − baseline)")
    print(f"Baseline: {baseline_label}")
    print("=" * 80)

    for rmsd_col in paired_rmsd_cols:
        if rmsd_col not in base_df.columns:
            continue
        base_scores_all = pd.to_numeric(base_df[rmsd_col], errors="coerce")
        base_lengths_all = base_df["L"] if "L" in base_df.columns else pd.Series(np.nan, index=base_df.index)

        print(f"\n── {rmsd_col} ──")
        header = (f"{'variant':<28}{'L':>6}{'n':>6}{'mean Δ':>10}{'median Δ':>11}"
                  f"{'%v<b':>8}{'wilc p':>10}{'des(b)':>9}{'des(v)':>9}{'mcN p':>9}")
        print(header)
        print("-" * len(header))

        for variant_label in present[1:]:
            var_df = dfs[variant_label].set_index("id_gen")
            if rmsd_col not in var_df.columns:
                continue
            common = base_df.index.intersection(var_df.index)
            if len(common) == 0:
                continue
            base_scores = base_scores_all.loc[common]
            var_scores = pd.to_numeric(var_df.loc[common, rmsd_col], errors="coerce")
            lengths = base_lengths_all.loc[common]

            # Per-length groups + Overall
            length_vals = sorted({int(x) for x in lengths.dropna().unique()})
            groups = [("L=" + str(l), lengths == l) for l in length_vals]
            groups.append(("Overall", pd.Series(True, index=common)))

            for gname, mask in groups:
                bs, vs = base_scores[mask], var_scores[mask]
                rmsd_stats = _paired_scrmsd_stats(bs, vs)
                des_stats = _paired_designability_stats(bs, vs)
                row = {"rmsd_col": rmsd_col, "variant": variant_label, "group": gname,
                       **rmsd_stats,
                       "base_designable": des_stats["base_designable"],
                       "variant_designable": des_stats["variant_designable"],
                       "discordant_v_only": des_stats["discordant_v_only"],
                       "discordant_b_only": des_stats["discordant_b_only"],
                       "mcnemar_p": des_stats["mcnemar_p"]}
                paired_rows.append(row)

                def _f(x, w, spec):
                    if isinstance(x, float) and math.isnan(x):
                        return f"{'nan':>{w}}"
                    return ("{:>" + spec + "}").format(x)

                if rmsd_stats["n_pairs"] == 0:
                    print(f"{variant_label[:27]:<28}{gname:>6}{0:>6d}    (no paired data)")
                else:
                    print(
                        f"{variant_label[:27]:<28}"
                        f"{gname:>6}"
                        f"{rmsd_stats['n_pairs']:>6d}"
                        f"{_f(rmsd_stats['mean_delta'], 10, '+10.3f')}"
                        f"{_f(rmsd_stats['median_delta'], 11, '+11.3f')}"
                        f"{_f(rmsd_stats['frac_variant_better'], 8, '8.2%')}"
                        f"{_f(rmsd_stats['wilcoxon_p'], 10, '10.4f')}"
                        f"{_f(des_stats['base_designable'], 9, '9.2%')}"
                        f"{_f(des_stats['variant_designable'], 9, '9.2%')}"
                        f"{_f(des_stats['mcnemar_p'], 9, '9.4f')}"
                    )

    if paired_rows:
        out_csv = os.path.join(os.path.dirname(entries[0][1]), "schedule_comparison_paired.csv")
        pd.DataFrame(paired_rows).to_csv(out_csv, index=False)
        print(f"\nPaired stats saved: {out_csv}")
elif len(present) >= 2 and paired_rmsd_cols:
    print("\n[!] Skipping paired analysis: 'id_gen' column missing in at least one CSV.")

# ── Plots ─────────────────────────────────────────────────────────────────────
rmsd_cols = sorted({
    c for df in dfs.values()
    for c in df.columns
    if c.startswith("_res_scRMSD_") or c.startswith("_res_co_scRMSD_")
})

if not rmsd_cols:
    print("No scRMSD columns found — skipping plots.")
    sys.exit(0)

# Use the path of the first entry for output dir
first_path = entries[0][1]

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

    n_labels = len(series)
    bar_w = 0.8 / max(n_labels, 1)
    offsets = np.linspace(-(n_labels - 1) * bar_w / 2, (n_labels - 1) * bar_w / 2, n_labels)
    x = np.arange(len(groups))

    fig, (ax_rmsd, ax_frac) = plt.subplots(1, 2, figsize=(max(10, 3 * len(groups)), 5),
                                            constrained_layout=True)

    for i, (label, s) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
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
        # Truncate label for legend if too long
        legend_label = label if len(label) <= 40 else label[:37] + "..."
        ax_rmsd.bar(xs, means_rmsd, bar_w * 0.9, yerr=stds_rmsd,
                    label=legend_label, color=color, capsize=4, alpha=0.85,
                    error_kw={"elinewidth": 1.2})
        ax_frac.bar(xs, means_frac, bar_w * 0.9, label=legend_label, color=color, alpha=0.85)

    col_short = rmsd_col.replace("_res_scRMSD_", "").replace("_", " ")

    ax_rmsd.axhline(2.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="2Å threshold")
    ax_rmsd.set_xticks(x); ax_rmsd.set_xticklabels(groups)
    ax_rmsd.set_xlabel("Protein length (residues)")
    ax_rmsd.set_ylabel("mean scRMSD (Å) ± std")
    ax_rmsd.set_title(f"scRMSD — {col_short}")
    ax_rmsd.legend(fontsize=7, loc="upper right")

    ax_frac.set_xticks(x); ax_frac.set_xticklabels(groups)
    ax_frac.set_xlabel("Protein length (residues)")
    ax_frac.set_ylabel("Fraction designable (scRMSD < 2Å)")
    ax_frac.set_title(f"Designability — {col_short}")
    ax_frac.set_ylim(0, 1.05)
    ax_frac.legend(fontsize=7, loc="upper right")

    out_png = os.path.join(
        os.path.dirname(first_path),
        f"schedule_comparison_{rmsd_col.replace('_res_scRMSD_', '')}.png"
    )
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out_png}")
