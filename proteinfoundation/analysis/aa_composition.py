"""Compare per-amino-acid composition: generated vs PDB.

Reads two FASTAs, length-filters the reference to match the generated
range, computes per-AA frequency (averaged across proteins so each
protein contributes equally), and prints/plots a sorted comparison.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AAS = list("ACDEFGHIKLMNPQRSTVWY")  # 20 canonical, alphabetical


def read_fasta(path: Path) -> list[tuple[str, str]]:
    seqs: list[tuple[str, str]] = []
    name, buf = None, []
    with path.open() as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    seqs.append((name, "".join(buf)))
                name = line[1:].split()[0]
                buf = []
            elif line:
                buf.append(line)
        if name is not None:
            seqs.append((name, "".join(buf)))
    return seqs


def per_protein_freq(seqs: list[str]) -> np.ndarray:
    """Return [N, 20] per-protein AA fractions. Each row sums to ~1
    (AAs outside the 20 canonical, like X, are dropped from numerator
    AND denominator, so the row reflects only canonical-AA composition)."""
    out = np.zeros((len(seqs), 20), dtype=np.float64)
    for i, s in enumerate(seqs):
        c = Counter(s)
        total = sum(c.get(a, 0) for a in AAS)
        if total == 0:
            continue
        for j, a in enumerate(AAS):
            out[i, j] = c.get(a, 0) / total
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path,
                    default=Path("results/generated_stratified_300_800/sequences.fasta"))
    ap.add_argument("--ref", type=Path,
                    default=Path("data/pdb_train/seq_df_pdb_f1_minl50_maxl500_mtprotein_etdiffractionEM_minoNone_maxoNone_minr0.0_maxr5.0_hl_rl_rnsrTrue_rpuFalse_l_rcuFalse_latents.fasta"))
    ap.add_argument("--length-min", type=int, default=300)
    ap.add_argument("--length-max", type=int, default=800)
    ap.add_argument("--out", type=Path,
                    default=Path("results/property_comparison/stratified_vs_pdb"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    gen = read_fasta(args.gen)
    ref = read_fasta(args.ref)
    print(f"[load] gen: {len(gen)} seqs from {args.gen.name}")
    print(f"[load] ref: {len(ref)} seqs from {args.ref.name}")

    gen = [(n, s) for n, s in gen if args.length_min <= len(s) <= args.length_max]
    ref = [(n, s) for n, s in ref if args.length_min <= len(s) <= args.length_max]
    print(f"[filter {args.length_min}-{args.length_max}]  gen={len(gen)}  ref={len(ref)}")

    gen_freq = per_protein_freq([s for _, s in gen])  # [Ng, 20]
    ref_freq = per_protein_freq([s for _, s in ref])  # [Nr, 20]

    df = pd.DataFrame({
        "aa":          AAS,
        "ref_mean":    ref_freq.mean(axis=0),
        "gen_mean":    gen_freq.mean(axis=0),
        "ref_sd":      ref_freq.std(axis=0, ddof=1),
        "gen_sd":      gen_freq.std(axis=0, ddof=1),
    })
    df["abs_diff"]    = df["gen_mean"] - df["ref_mean"]
    df["rel_diff_pct"] = 100.0 * df["abs_diff"] / df["ref_mean"]

    # sort by gen frequency (so the table reads "what does the model love")
    df_by_gen = df.sort_values("gen_mean", ascending=False).reset_index(drop=True)
    print("\n=== sorted by GEN frequency ===")
    print(df_by_gen.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    # most over-/under-represented
    df_by_diff = df.sort_values("abs_diff", ascending=False).reset_index(drop=True)
    print("\n=== sorted by GEN − REF (most over-represented at top) ===")
    print(df_by_diff.to_string(index=False, float_format=lambda v: f"{v:+7.4f}"))

    # write CSV
    csv_path = args.out / "aa_composition.csv"
    df_by_gen.to_csv(csv_path, index=False)
    print(f"\n[write] {csv_path}")

    # plot 1: side-by-side bar chart
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
    order = df.sort_values("ref_mean", ascending=False)["aa"].tolist()
    df_sorted = df.set_index("aa").loc[order].reset_index()
    x = np.arange(20)
    w = 0.4
    axes[0].bar(x - w/2, df_sorted["ref_mean"], w, label="PDB (300-800)",
                color="#1f77b4", yerr=df_sorted["ref_sd"], capsize=2,
                error_kw={"alpha": 0.4})
    axes[0].bar(x + w/2, df_sorted["gen_mean"], w, label="generated",
                color="#d62728", yerr=df_sorted["gen_sd"], capsize=2,
                error_kw={"alpha": 0.4})
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_sorted["aa"])
    axes[0].set_ylabel("mean per-protein fraction")
    axes[0].set_title("AA composition: generated vs PDB (sorted by PDB freq)")
    axes[0].legend()

    # plot 2: relative deviation (gen - ref) / ref, with sign coloring
    diff_sorted = df_sorted["gen_mean"] - df_sorted["ref_mean"]
    rel = 100.0 * diff_sorted / df_sorted["ref_mean"]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in rel]
    axes[1].bar(x, rel, color=colors)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_sorted["aa"])
    axes[1].set_ylabel("(gen − ref) / ref   [%]")
    axes[1].set_title("Per-AA over-/under-representation (red = generated has more)")

    fig.tight_layout()
    png_path = args.out / "aa_composition.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {png_path}")


if __name__ == "__main__":
    main()
