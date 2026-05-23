"""Compare fixt1 (`fixt1_full_replication_2026_05_18`) vs NA-v1 Pareto frontiers.

Pulls per-cell `properties_guided.csv` and `codesign_guided.csv` for both
predictor variants, anchors them to the *same* paired-by-seed unsteered
baseline, and prints a side-by-side σ-delivery + codesign table.

For each (direction, w):
  - codesign rate from `codesign_guided.csv` (coScRMSD_ca < 2.0)
  - target-property mean from `properties_guided.csv`
  - σ-delivery = want_sign · (mean_steered − μ_unsteered) / σ_unsteered
    where (μ, σ) come from `generated_stratified_300_800_nsteps400/properties_generated.csv`
    filtered to L∈[290, 510] (n=422), same anchor as E066/E067/E068.

Combo direction reports `aggregate_sigma = σ_SWI + (−1)·σ_TANGO`.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ks2218/la-proteina")
FIXT1_TREE = ROOT / "results/fixt1_full_replication_2026_05_18"
UNSTEERED_PROPS = ROOT / "results/generated_stratified_300_800_nsteps400/properties_generated.csv"
UNSTEERED_CODESIGN = ROOT / "results/noise_aware_ensemble_sweep/codesign_unsteered_matched_seed.csv"

# (direction, w) -> (fixt1_cell_dir, nav1_cell_dir)
NAV1_ROOT = {
    "camsol_max":         ROOT / "results/noise_aware_high_w_scout",
    "tango_min":          ROOT / "results/noise_aware_high_w_scout",
    "iupred_max":         ROOT / "results/iupred_max_scout",
    "combo_camsol_tango": ROOT / "results/combo_camsol_tango_scout",
}
FIXT1_ROOT = {
    "camsol_max":         FIXT1_TREE / "E066_high_w",
    "tango_min":          FIXT1_TREE / "E066_high_w",
    "iupred_max":         FIXT1_TREE / "E067_iupred_max",
    "combo_camsol_tango": FIXT1_TREE / "E068_combo_camsol_tango",
}
DIRECTIONS = {
    "camsol_max":         ("swi",                          +1, [32, 48, 64, 128]),
    "tango_min":          ("tango_total",                  -1, [32, 48, 64, 128]),
    "iupred_max":         ("iupred3_fraction_disordered",  +1, [32,     64, 128]),
    "combo_camsol_tango": ("__combo__",                    +1, [32, 48, 64, 128]),
}


def unsteered_stats() -> dict[str, tuple[float, float]]:
    df = pd.read_csv(UNSTEERED_PROPS)
    df = df[df.sequence_length.between(290, 510)]
    out = {}
    for col in ["swi", "tango_total", "iupred3_fraction_disordered", "net_charge_ph7"]:
        if col in df:
            out[col] = (float(df[col].mean()), float(df[col].std()))
    out["__n_baseline__"] = (len(df), 0)
    return out


def codesign_rate(csv_path: Path) -> tuple[float, int]:
    if not csv_path.exists():
        return (float("nan"), 0)
    df = pd.read_csv(csv_path)
    if "coScRMSD_ca" not in df:
        return (float("nan"), 0)
    finite = df.loc[np.isfinite(df.coScRMSD_ca), "coScRMSD_ca"]
    if not len(finite):
        return (float("nan"), 0)
    return (float((finite < 2.0).mean()), len(finite))


def sigma_delivery(csv_path: Path, target_col: str, sign: int,
                   stats: dict) -> tuple[float, int]:
    if not csv_path.exists():
        return (float("nan"), 0)
    df = pd.read_csv(csv_path)
    if target_col == "__combo__":
        if "swi" not in df or "tango_total" not in df:
            return (float("nan"), 0)
        mu_swi, sd_swi = stats["swi"]
        mu_tan, sd_tan = stats["tango_total"]
        z_swi = (df["swi"].mean() - mu_swi) / sd_swi
        z_tan = (df["tango_total"].mean() - mu_tan) / sd_tan
        return (float(z_swi - z_tan), int(df["swi"].notna().sum()))
    if target_col not in df:
        return (float("nan"), 0)
    mu, sd = stats.get(target_col, (float("nan"), float("nan")))
    if sd <= 0 or math.isnan(sd):
        return (float("nan"), 0)
    z = (df[target_col].mean() - mu) / sd
    return (float(sign * z), int(df[target_col].notna().sum()))


def main():
    stats = unsteered_stats()
    print(f"Unsteered baseline: n={stats['__n_baseline__'][0]} proteins at L∈[290, 510]")
    for col in ["swi", "tango_total", "iupred3_fraction_disordered"]:
        m, s = stats[col]
        print(f"  {col}: μ={m:.4f}, σ={s:.4f}")
    print()

    rows = []
    # Unsteered baseline anchor row — codesign from the paired-seed n=30 CSV
    # (seeds 42-51 × L=300/400/500), property σ-delivery is 0 by construction.
    base_codes, base_n = codesign_rate(UNSTEERED_CODESIGN)
    rows.append({
        "direction": "unsteered",
        "w": 0,
        "fixt1_codesign_pct": base_codes * 100,
        "fixt1_n_codes": base_n,
        "nav1_codesign_pct":  base_codes * 100,
        "nav1_n_codes":  base_n,
        "d_codesign_pp":      0.0,
        "fixt1_sigma":  0.0,
        "fixt1_n_prop": stats["__n_baseline__"][0],
        "nav1_sigma":   0.0,
        "nav1_n_prop":  stats["__n_baseline__"][0],
        "d_sigma":      0.0,
    })

    for direction, (target, sign, w_list) in DIRECTIONS.items():
        for w in w_list:
            cell_name = f"{direction}_w{w}"
            fixt1_cell = FIXT1_ROOT[direction] / cell_name
            nav1_cell  = NAV1_ROOT[direction]  / cell_name

            f_codes, f_n_codes = codesign_rate(fixt1_cell / "codesign_guided.csv")
            n_codes, n_n_codes = codesign_rate(nav1_cell  / "codesign_guided.csv")
            f_sig, f_n_prop = sigma_delivery(fixt1_cell / "properties_guided.csv",
                                             target, sign, stats)
            n_sig, n_n_prop = sigma_delivery(nav1_cell  / "properties_guided.csv",
                                             target, sign, stats)

            rows.append({
                "direction": direction,
                "w": w,
                "fixt1_codesign_pct": f_codes * 100 if not math.isnan(f_codes) else float("nan"),
                "fixt1_n_codes": f_n_codes,
                "nav1_codesign_pct":  n_codes * 100 if not math.isnan(n_codes) else float("nan"),
                "nav1_n_codes":  n_n_codes,
                "d_codesign_pp":      (f_codes - n_codes) * 100 if not (math.isnan(f_codes) or math.isnan(n_codes)) else float("nan"),
                "fixt1_sigma":  f_sig,
                "fixt1_n_prop": f_n_prop,
                "nav1_sigma":   n_sig,
                "nav1_n_prop":  n_n_prop,
                "d_sigma":      f_sig - n_sig if not (math.isnan(f_sig) or math.isnan(n_sig)) else float("nan"),
            })

    df = pd.DataFrame(rows)
    out = ROOT / "results/fixt1_full_replication_2026_05_18/fixt1_vs_nav1_pareto.csv"
    df.to_csv(out, index=False)

    fmt = {
        "fixt1_codesign_pct": "{:5.1f}".format,
        "nav1_codesign_pct":  "{:5.1f}".format,
        "d_codesign_pp":      "{:+5.1f}".format,
        "fixt1_sigma":        "{:+5.2f}".format,
        "nav1_sigma":         "{:+5.2f}".format,
        "d_sigma":            "{:+5.2f}".format,
    }
    print("\n=== fixt1 vs NA-v1 — Pareto frontier (anchored to L∈[290,510] unsteered baseline) ===")
    cols = ["direction", "w",
            "fixt1_codesign_pct", "nav1_codesign_pct", "d_codesign_pp",
            "fixt1_sigma",        "nav1_sigma",        "d_sigma",
            "fixt1_n_prop",       "nav1_n_prop"]
    print(df[cols].to_string(index=False, formatters=fmt))

    # Pooled summary (excludes the unsteered baseline row)
    steered = df[df.direction != "unsteered"]
    print("\n=== Pooled (mean across direction & w, equal-weight cells; baseline excluded) ===")
    for c in ["d_codesign_pp", "d_sigma"]:
        mean = steered[c].mean()
        wins_fixt1 = (steered[c] > 0).sum()
        ties = (steered[c].abs() < 1e-9).sum()
        losses = (steered[c] < 0).sum()
        print(f"  {c:18s}  mean={mean:+.2f}   fixt1>NA: {wins_fixt1}, =: {ties}, <: {losses}  (n={steered[c].notna().sum()})")

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
