"""Compare developability-property distributions between two CSVs.

Use this to sanity-check a freshly generated set against the training-set
reference panel: per-property summary stats (mean / sd / var / skew /
kurtosis / IQR), KS + Wasserstein distance, modality count from a KDE,
and a grid of overlaid histogram+KDE plots.

The two CSV schemas don't agree perfectly (training-set panel uses
`net_charge` / `iupred3` / `tango` / `sap` / `rg`; generated panel uses
`net_charge_ph7` / `iupred3_mean` / `tango_total` / `sap_total` /
`radius_of_gyration`), so comparison happens on a canonical name space
defined by `COLUMN_MAP`.

Usage (from repo root):
    python proteinfoundation/analysis/compare_properties.py \\
        --ref laproteina_steerability/data/properties.csv \\
        --gen results/generated_stratified_300_800/properties_generated.csv \\
        --out results/property_comparison/stratified_vs_pdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks


# Canonical name -> (ref-csv column, gen-csv column).
# Length is included so its distribution is also reported (it's a sanity
# check on stratified sampling, not a developability property).
COLUMN_MAP: dict[str, tuple[str, str]] = {
    "sequence_length":              ("sequence_length",              "sequence_length"),
    "swi":                          ("swi",                          "swi"),
    "tango":                        ("tango",                        "tango_total"),
    "tango_aggregation_positions":  ("tango_aggregation_positions",  "tango_aggregation_positions"),
    "net_charge":                   ("net_charge",                   "net_charge_ph7"),
    "pI":                           ("pI",                           "pI"),
    "iupred3":                      ("iupred3",                      "iupred3_mean"),
    "iupred3_fraction_disordered":  ("iupred3_fraction_disordered",  "iupred3_fraction_disordered"),
    "shannon_entropy":              ("shannon_entropy",              "shannon_entropy"),
    "hydrophobic_patch_total_area": ("hydrophobic_patch_total_area", "hydrophobic_patch_total_area"),
    "hydrophobic_patch_n_large":    ("hydrophobic_patch_n_large",    "hydrophobic_patch_n_large"),
    "sap":                          ("sap",                          "sap_total"),
    "scm_positive":                 ("scm_positive",                 "scm_positive"),
    "scm_negative":                 ("scm_negative",                 "scm_negative"),
    "rg":                           ("rg",                           "radius_of_gyration"),
}


def load(path: Path, mapping_idx: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {COLUMN_MAP[k][mapping_idx]: k for k in COLUMN_MAP if COLUMN_MAP[k][mapping_idx] in df.columns}
    df = df.rename(columns=rename)
    keep = [k for k in COLUMN_MAP if k in df.columns]
    return df[keep].copy()


def kde_modality(x: np.ndarray, grid: int = 512, prominence_frac: float = 0.05) -> int:
    """Count peaks in a Gaussian KDE of `x`. Prominence threshold is a
    fraction of the max density so tiny ripples in long tails don't get
    counted as modes."""
    x = x[np.isfinite(x)]
    if x.size < 5 or np.allclose(x, x[0]):
        return 1
    try:
        kde = stats.gaussian_kde(x)
    except (np.linalg.LinAlgError, ValueError):
        return 1
    lo, hi = np.min(x), np.max(x)
    pad = 0.05 * (hi - lo + 1e-9)
    xs = np.linspace(lo - pad, hi + pad, grid)
    ys = kde(xs)
    peaks, _ = find_peaks(ys, prominence=prominence_frac * ys.max())
    return max(1, len(peaks))


def per_property_stats(s: pd.Series) -> dict:
    s = s.dropna().to_numpy()
    if s.size == 0:
        return {k: np.nan for k in
                ["n", "mean", "sd", "var", "median", "iqr", "skew",
                 "kurt", "min", "max", "modes"]}
    q25, q75 = np.percentile(s, [25, 75])
    return {
        "n":      s.size,
        "mean":   float(np.mean(s)),
        "sd":     float(np.std(s, ddof=1)),
        "var":    float(np.var(s, ddof=1)),
        "median": float(np.median(s)),
        "iqr":    float(q75 - q25),
        "skew":   float(stats.skew(s)),
        "kurt":   float(stats.kurtosis(s)),  # excess kurtosis
        "min":    float(np.min(s)),
        "max":    float(np.max(s)),
        "modes":  kde_modality(s),
    }


def compare_pair(ref: np.ndarray, gen: np.ndarray) -> dict:
    """Distance metrics between the two distributions."""
    ref = ref[np.isfinite(ref)]
    gen = gen[np.isfinite(gen)]
    if ref.size == 0 or gen.size == 0:
        return {"ks_d": np.nan, "ks_p": np.nan, "wasserstein": np.nan,
                "mean_diff": np.nan, "std_mean_diff": np.nan}
    ks = stats.ks_2samp(ref, gen, alternative="two-sided", mode="auto")
    pooled_sd = np.sqrt(((ref.size - 1) * np.var(ref, ddof=1) +
                         (gen.size - 1) * np.var(gen, ddof=1)) /
                        (ref.size + gen.size - 2))
    mean_diff = float(np.mean(gen) - np.mean(ref))
    return {
        "ks_d":          float(ks.statistic),
        "ks_p":          float(ks.pvalue),
        "wasserstein":   float(stats.wasserstein_distance(ref, gen)),
        "mean_diff":     mean_diff,
        # gen mean expressed in reference-pooled standard deviations
        # (Cohen's d using pooled SD; not Glass's delta).
        "std_mean_diff": float(mean_diff / pooled_sd) if pooled_sd > 0 else np.nan,
    }


def plot_grid(ref: pd.DataFrame, gen: pd.DataFrame, props: list[str],
              out_path: Path, ref_label: str, gen_label: str) -> None:
    n = len(props)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, prop in zip(axes, props):
        ref_x = ref[prop].dropna().to_numpy()
        gen_x = gen[prop].dropna().to_numpy()
        if ref_x.size == 0 and gen_x.size == 0:
            ax.set_visible(False)
            continue
        # shared bin range for honest visual comparison
        lo = np.nanmin([ref_x.min() if ref_x.size else np.inf,
                        gen_x.min() if gen_x.size else np.inf])
        hi = np.nanmax([ref_x.max() if ref_x.size else -np.inf,
                        gen_x.max() if gen_x.size else -np.inf])
        bins = np.linspace(lo, hi, 50) if hi > lo else 10
        if ref_x.size:
            ax.hist(ref_x, bins=bins, density=True, alpha=0.45,
                    color="#1f77b4", label=ref_label)
        if gen_x.size:
            ax.hist(gen_x, bins=bins, density=True, alpha=0.45,
                    color="#d62728", label=gen_label)
        for x, color in [(ref_x, "#1f77b4"), (gen_x, "#d62728")]:
            if x.size >= 5 and not np.allclose(x, x[0]):
                try:
                    kde = stats.gaussian_kde(x)
                    xs = np.linspace(lo, hi, 256)
                    ax.plot(xs, kde(xs), color=color, lw=1.4)
                except (np.linalg.LinAlgError, ValueError):
                    pass
        ax.set_title(prop, fontsize=10)
        ax.tick_params(labelsize=8)
    # one legend for the figure
    axes[0].legend(fontsize=8, loc="best")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{gen_label}  vs  {ref_label}", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path,
                    default=Path("laproteina_steerability/data/properties.csv"),
                    help="Reference (training-set) developability panel CSV.")
    ap.add_argument("--gen", type=Path,
                    default=Path("results/generated_stratified_300_800/properties_generated.csv"),
                    help="Generated developability panel CSV.")
    ap.add_argument("--out", type=Path,
                    default=Path("results/property_comparison/latest"),
                    help="Output directory (created if missing).")
    ap.add_argument("--ref-label", default="train (PDB)")
    ap.add_argument("--gen-label", default="generated")
    ap.add_argument("--length-min", type=int, default=None,
                    help="Optional: restrict reference to this min sequence_length.")
    ap.add_argument("--length-max", type=int, default=None,
                    help="Optional: restrict reference to this max sequence_length.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    ref = load(args.ref, mapping_idx=0)
    gen = load(args.gen, mapping_idx=1)

    if args.length_min is not None:
        ref = ref[ref["sequence_length"] >= args.length_min]
    if args.length_max is not None:
        ref = ref[ref["sequence_length"] <= args.length_max]

    common = [k for k in COLUMN_MAP if k in ref.columns and k in gen.columns]
    print(f"[load] ref n={len(ref)}  gen n={len(gen)}  common cols={len(common)}")
    print(f"[load] common: {common}\n")

    rows = []
    for prop in common:
        r = per_property_stats(ref[prop])
        g = per_property_stats(gen[prop])
        d = compare_pair(ref[prop].to_numpy(), gen[prop].to_numpy())
        rows.append({
            "property":      prop,
            "ref_n":         r["n"],   "gen_n":       g["n"],
            "ref_mean":      r["mean"], "gen_mean":   g["mean"],
            "ref_sd":        r["sd"],   "gen_sd":     g["sd"],
            "ref_var":       r["var"],  "gen_var":    g["var"],
            "ref_median":    r["median"], "gen_median": g["median"],
            "ref_iqr":       r["iqr"],  "gen_iqr":    g["iqr"],
            "ref_skew":      r["skew"], "gen_skew":   g["skew"],
            "ref_kurt":      r["kurt"], "gen_kurt":   g["kurt"],
            "ref_min":       r["min"],  "gen_min":    g["min"],
            "ref_max":       r["max"],  "gen_max":    g["max"],
            "ref_modes":     r["modes"], "gen_modes": g["modes"],
            "mean_diff":     d["mean_diff"],
            "mean_diff_sd":  d["std_mean_diff"],
            "ks_d":          d["ks_d"],
            "ks_p":          d["ks_p"],
            "wasserstein":   d["wasserstein"],
        })
    summary = pd.DataFrame(rows)

    csv_path = args.out / "summary.csv"
    summary.to_csv(csv_path, index=False)

    # short, human-readable view
    short = summary[[
        "property", "ref_n", "gen_n",
        "ref_mean", "gen_mean", "mean_diff_sd",
        "ref_sd", "gen_sd",
        "ref_modes", "gen_modes",
        "ks_d", "wasserstein",
    ]].copy()
    fmt = {
        "ref_mean": "{:.3g}", "gen_mean": "{:.3g}", "mean_diff_sd": "{:+.2f}",
        "ref_sd": "{:.3g}", "gen_sd": "{:.3g}",
        "ks_d": "{:.3f}", "wasserstein": "{:.3g}",
    }
    for col, f in fmt.items():
        short[col] = short[col].map(lambda v, f=f: f.format(v) if pd.notna(v) else "—")
    print(short.to_string(index=False))
    print()
    print("Legend:")
    print("  mean_diff_sd : (gen_mean − ref_mean) / pooled_sd  (Cohen's d)")
    print("  ks_d         : 2-sample KS statistic, larger = more different")
    print("  wasserstein  : Earth-mover's distance in property units")
    print("  *_modes      : peak count in a Gaussian KDE (rough modality)")

    # plots
    plot_props = [p for p in common if p != "sequence_length"] + ["sequence_length"]
    plot_grid(ref, gen, plot_props,
              args.out / "distributions.png",
              args.ref_label, args.gen_label)

    print(f"\n[write] {csv_path}")
    print(f"[write] {args.out / 'distributions.png'}")


if __name__ == "__main__":
    main()
