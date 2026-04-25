"""
Cheap diagnostics for the PDB-vs-generated property comparison experiment.

Computes:
  1. Count of generated proteins per 50-residue bin in [300, 800).
  2. Per-bin counts and overall summary for the PDB property table (same bins).
  3. KS distance between the two length distributions and per-bin counts table.
  4. 14x14 Pearson + Spearman correlation matrices on the PDB property table.
  5. Li-Ji (2005) effective number of independent tests M_eff for both matrices.
  6. Same M_eff stratified by length bin (do correlations change with length?).

Outputs to /home/ks2218/la-proteina/analysis_cheap_diagnostics/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr

ROOT = Path("/home/ks2218/la-proteina")
OUT = ROOT / "analysis_cheap_diagnostics"
OUT.mkdir(exist_ok=True)

PDB_CSV = ROOT / "laproteina_steerability/data/properties.csv"
GEN_CSV = ROOT / "results/generated_baseline_300_800/properties_generated.csv"

# Mapping: PDB column name -> generated column name (the 14 shared properties).
PROPERTY_PAIRS = [
    ("swi", "swi"),
    ("tango", "tango_total"),
    ("tango_aggregation_positions", "tango_aggregation_positions"),
    ("net_charge", "net_charge_ph7"),
    ("pI", "pI"),
    ("iupred3", "iupred3_mean"),
    ("iupred3_fraction_disordered", "iupred3_fraction_disordered"),
    ("shannon_entropy", "shannon_entropy"),
    ("hydrophobic_patch_total_area", "hydrophobic_patch_total_area"),
    ("hydrophobic_patch_n_large", "hydrophobic_patch_n_large"),
    ("sap", "sap_total"),
    ("scm_positive", "scm_positive"),
    ("scm_negative", "scm_negative"),
    ("rg", "radius_of_gyration"),
]
PDB_COLS = [a for a, _ in PROPERTY_PAIRS]
GEN_COLS = [b for _, b in PROPERTY_PAIRS]
CANON = PDB_COLS  # canonical names used for output

BIN_EDGES = list(range(300, 801, 50))  # [300,350,...,800]
BIN_LABELS = [f"[{lo},{hi})" for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]


def li_ji_meff(corr: np.ndarray) -> float:
    """Li & Ji (2005) effective number of independent tests.

    M_eff = sum_i [ I(lambda_i >= 1) + (lambda_i - floor(lambda_i)) ]
    where lambda_i are eigenvalues of the correlation matrix.
    """
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.clip(eigvals, 0.0, None)  # numerical safety
    m_eff = float(np.sum((eigvals >= 1).astype(float) + (eigvals - np.floor(eigvals))))
    return m_eff


def write_md(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------
    print("Loading PDB properties ...")
    pdb = pd.read_csv(PDB_CSV)
    print(f"  {len(pdb)} PDB rows, {pdb['sequence_length'].notna().sum()} with length")

    print("Loading generated properties ...")
    gen = pd.read_csv(GEN_CSV)
    print(f"  {len(gen)} generated rows")

    # -----------------------------------------------------------------------
    # 1+2. Per-bin counts
    # -----------------------------------------------------------------------
    pdb["bin"] = pd.cut(pdb["sequence_length"], BIN_EDGES, right=False, labels=BIN_LABELS)
    gen["bin"] = pd.cut(gen["sequence_length"], BIN_EDGES, right=False, labels=BIN_LABELS)

    bin_counts = pd.DataFrame({
        "bin": BIN_LABELS,
        "pdb_n": [int((pdb["bin"] == b).sum()) for b in BIN_LABELS],
        "generated_n": [int((gen["bin"] == b).sum()) for b in BIN_LABELS],
    })
    bin_counts.to_csv(OUT / "length_bin_counts.csv", index=False)
    print("\nPer-bin counts (300-800 in 50-residue bins):")
    print(bin_counts.to_string(index=False))

    # -----------------------------------------------------------------------
    # 3. KS distance on length distributions (300-800 only)
    # -----------------------------------------------------------------------
    pdb_in_range = pdb["sequence_length"].between(300, 800, inclusive="left")
    gen_in_range = gen["sequence_length"].between(300, 800, inclusive="left")
    ks_stat, ks_p = ks_2samp(
        pdb.loc[pdb_in_range, "sequence_length"].values,
        gen.loc[gen_in_range, "sequence_length"].values,
    )
    print(f"\nKS distance on length (300-800 only): D = {ks_stat:.4f}, p = {ks_p:.3e}")
    print(f"  PDB in range: {int(pdb_in_range.sum())}, Generated in range: {int(gen_in_range.sum())}")

    # -----------------------------------------------------------------------
    # 4. Property correlation matrices (Pearson + Spearman) on PDB
    # -----------------------------------------------------------------------
    pdb_props = pdb[PDB_COLS].copy()
    nan_pre = pdb_props.isna().sum()
    pdb_props = pdb_props.dropna()
    print(f"\nDropping NaN rows for correlation: {len(pdb)} -> {len(pdb_props)}")
    print("Per-property NaN counts (pre-drop, 14 properties):")
    print(nan_pre.to_string())

    pearson = pdb_props.corr(method="pearson")
    spearman = pdb_props.corr(method="spearman")
    pearson.to_csv(OUT / "pdb_pearson_corr.csv")
    spearman.to_csv(OUT / "pdb_spearman_corr.csv")

    # Highest absolute off-diagonal pairs
    def top_pairs(corr_mat, k=10):
        rows = []
        m = corr_mat.values
        n = m.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                rows.append((corr_mat.index[i], corr_mat.columns[j], m[i, j]))
        rows.sort(key=lambda x: abs(x[2]), reverse=True)
        return rows[:k]

    print("\nTop 10 Pearson correlations (|r|):")
    for a, b, r in top_pairs(pearson, 10):
        print(f"  {a:35s}  {b:35s}  {r:+.3f}")

    print("\nTop 10 Spearman correlations (|rho|):")
    for a, b, r in top_pairs(spearman, 10):
        print(f"  {a:35s}  {b:35s}  {r:+.3f}")

    # -----------------------------------------------------------------------
    # 5. Li-Ji effective number of independent tests
    # -----------------------------------------------------------------------
    m_eff_pearson = li_ji_meff(pearson.values)
    m_eff_spearman = li_ji_meff(spearman.values)
    eigvals_p = np.sort(np.linalg.eigvalsh(pearson.values))[::-1]
    eigvals_s = np.sort(np.linalg.eigvalsh(spearman.values))[::-1]
    print(f"\nLi-Ji M_eff (Pearson):  {m_eff_pearson:.3f}  (out of {len(PDB_COLS)})")
    print(f"Li-Ji M_eff (Spearman): {m_eff_spearman:.3f}  (out of {len(PDB_COLS)})")
    print(f"  Pearson  eigenvalues: {np.round(eigvals_p, 3).tolist()}")
    print(f"  Spearman eigenvalues: {np.round(eigvals_s, 3).tolist()}")

    # Bonferroni equivalents
    alpha = 0.05
    print(f"\nBonferroni @ alpha={alpha}:")
    print(f"  naive (14 tests):                 {alpha/14:.5f}")
    print(f"  Li-Ji Pearson  ({m_eff_pearson:.2f} tests):  {alpha/m_eff_pearson:.5f}")
    print(f"  Li-Ji Spearman ({m_eff_spearman:.2f} tests): {alpha/m_eff_spearman:.5f}")

    # -----------------------------------------------------------------------
    # 6. Per-bin correlation stability
    # -----------------------------------------------------------------------
    print("\nLi-Ji M_eff per length bin (Spearman, on PDB only):")
    rows = []
    for b in BIN_LABELS:
        sub = pdb.loc[pdb["bin"] == b, PDB_COLS].dropna()
        if len(sub) < 100:
            rows.append((b, len(sub), float("nan")))
            continue
        c = sub.corr(method="spearman").values
        rows.append((b, len(sub), li_ji_meff(c)))
    df_bins = pd.DataFrame(rows, columns=["bin", "n", "m_eff_spearman"])
    df_bins.to_csv(OUT / "li_ji_per_bin.csv", index=False)
    print(df_bins.to_string(index=False))

    # -----------------------------------------------------------------------
    # Summary markdown
    # -----------------------------------------------------------------------
    md = []
    md.append("# Cheap diagnostics — Exp 1, steps 1+2")
    md.append("")
    md.append(f"PDB property file: `{PDB_CSV}` ({len(pdb)} rows)")
    md.append(f"Generated property file: `{GEN_CSV}` ({len(gen)} rows)")
    md.append("")
    md.append("## Per-bin counts (50-residue bins, 300-800)")
    md.append("")
    md.append(df_to_md(bin_counts))
    md.append("")
    md.append(f"**KS distance on length (300-800 only):** D = {ks_stat:.4f}, p = {ks_p:.3e}")
    md.append(f"  PDB in range: {int(pdb_in_range.sum())}, Generated in range: {int(gen_in_range.sum())}")
    md.append("")
    md.append("## Property correlation matrix on PDB")
    md.append("")
    md.append(f"After NaN-drop: {len(pdb_props)} / {len(pdb)} PDB rows used.")
    md.append("")
    md.append(f"**Li-Ji M_eff (Pearson):  {m_eff_pearson:.3f}** (out of 14 properties)")
    md.append(f"**Li-Ji M_eff (Spearman): {m_eff_spearman:.3f}** (out of 14 properties)")
    md.append("")
    md.append(f"Pearson  eigenvalues (sorted desc): {np.round(eigvals_p, 3).tolist()}")
    md.append("")
    md.append(f"Spearman eigenvalues (sorted desc): {np.round(eigvals_s, 3).tolist()}")
    md.append("")
    md.append("### Top 10 |Pearson| pairs")
    md.append("")
    md.append("| prop A | prop B | r |")
    md.append("|---|---|---|")
    for a, b, r in top_pairs(pearson, 10):
        md.append(f"| {a} | {b} | {r:+.3f} |")
    md.append("")
    md.append("### Top 10 |Spearman| pairs")
    md.append("")
    md.append("| prop A | prop B | rho |")
    md.append("|---|---|---|")
    for a, b, r in top_pairs(spearman, 10):
        md.append(f"| {a} | {b} | {r:+.3f} |")
    md.append("")
    md.append("### Bonferroni thresholds at alpha=0.05")
    md.append("")
    md.append(f"- naive (14 tests): {alpha/14:.5f}")
    md.append(f"- Li-Ji Pearson  ({m_eff_pearson:.2f} tests): {alpha/m_eff_pearson:.5f}")
    md.append(f"- Li-Ji Spearman ({m_eff_spearman:.2f} tests): {alpha/m_eff_spearman:.5f}")
    md.append("")
    md.append("## Li-Ji M_eff per length bin (Spearman, PDB)")
    md.append("")
    md.append(df_to_md(df_bins))
    write_md(OUT / "summary.md", md)

    print(f"\nDone. Outputs in {OUT}/")


if __name__ == "__main__":
    main()
