"""Compare thermal-stability proxies between generated and PDB sequences.

Tier 1 — sequence-only proxies (always run, instant):
    - aliphatic_index   (Ikai 1980)            higher = more thermostable
    - ivywrel_fraction  (Zeldovich 2007)        higher = more thermostable
    - gravy             (Kyte-Doolittle)        higher = more hydrophobic
    - charged_fraction  (D+E+K+R)               higher = more charged
    - acidic_basic_ratio  (D+E)/(K+R)           >1 = net acidic

Tier 2 — TemStaPro ML predictor (opt-in, GPU recommended):
    Stand-in for DeepStabP, which only exists as a hosted web service and
    has no published offline weights. TemStaPro uses the same architecture
    family (mean-pooled ProtT5 embedding -> MLP head) and outputs
    probability that Tm > T for T in {40, 45, 50, 55, 60, 65}°C plus an
    aggregated thermophilicity label. Reference: Pudziuvelyte et al.,
    Bioinformatics 2024.

Usage:
    # Tier 1 only:
    python proteinfoundation/analysis/thermal_stability.py \\
        --gen results/generated_stratified_300_800/sequences.fasta \\
        --ref pdb_cluster_all_seqs.fasta \\
        --out results/thermal_stability/stratified_vs_pdb

    # With TemStaPro (one-time setup: pip install sentencepiece; clone
    # https://github.com/ievapudz/TemStaPro to a persistent path):
    python proteinfoundation/analysis/thermal_stability.py \\
        --temstapro-dir /home/ks2218/TemStaPro \\
        --temstapro-emb-dir /tmp/temstapro_emb
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------
# Tier 1: sequence proxies
# ----------------------------------------------------------------------

# Kyte-Doolittle hydropathy.
KD: dict[str, float] = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}
IVYWREL = set("IVYWREL")
ACIDIC = set("DE")
BASIC = set("KR")
AROMATIC = set("FWY")


def _composition(seq: str) -> dict[str, float]:
    seq = "".join(c for c in seq.upper() if c in KD)
    if not seq:
        return {}
    n = len(seq)
    return {a: seq.count(a) / n for a in KD}


def aliphatic_index(seq: str) -> float:
    """Ikai 1980. Mole-percent units (i.e. *100 fractions). Range typically 60-110."""
    f = _composition(seq)
    if not f:
        return float("nan")
    return 100.0 * (f["A"] + 2.9 * f["V"] + 3.9 * (f["I"] + f["L"]))


def ivywrel_fraction(seq: str) -> float:
    f = _composition(seq)
    return float("nan") if not f else sum(f[a] for a in IVYWREL)


def gravy(seq: str) -> float:
    f = _composition(seq)
    return float("nan") if not f else sum(f[a] * KD[a] for a in f)


def charged_fraction(seq: str) -> float:
    f = _composition(seq)
    return float("nan") if not f else sum(f[a] for a in ACIDIC | BASIC)


def log_acidic_basic_ratio(seq: str) -> float:
    """log10((D+E+pseudo)/(K+R+pseudo)). Symmetric, finite, no div-by-zero.
    Pseudocount = 1/L so zero-K+R proteins are heavy-tail negative, not inf."""
    f = _composition(seq)
    if not f:
        return float("nan")
    L = max(1, sum(seq.count(a) for a in KD))
    pseudo = 1.0 / L
    acidic = sum(f[a] for a in ACIDIC) + pseudo
    basic = sum(f[a] for a in BASIC) + pseudo
    return float(np.log10(acidic / basic))


def aromatic_fraction(seq: str) -> float:
    """F+W+Y mole fraction — buried-aromatic-core proxy."""
    f = _composition(seq)
    return float("nan") if not f else sum(f[a] for a in AROMATIC)


PROXY_FNS = {
    "aliphatic_index":         aliphatic_index,
    "ivywrel_fraction":        ivywrel_fraction,
    "gravy":                   gravy,
    "charged_fraction":        charged_fraction,
    "log_acidic_basic_ratio":  log_acidic_basic_ratio,
    "aromatic_fraction":       aromatic_fraction,
}


# ----------------------------------------------------------------------
# FASTA I/O
# ----------------------------------------------------------------------

def read_fasta(path: Path) -> list[tuple[str, str]]:
    out, name, buf = [], None, []
    with path.open() as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    out.append((name, "".join(buf)))
                name, buf = line[1:].split()[0], []
            elif line:
                buf.append(line)
        if name is not None:
            out.append((name, "".join(buf)))
    # drop duplicate-header / empty entries
    seen = set()
    deduped = []
    for n, s in out:
        if not s or n in seen:
            continue
        seen.add(n)
        deduped.append((n, s))
    return deduped


def tier1_table(fasta: Path,
                length_min: int | None,
                length_max: int | None) -> pd.DataFrame:
    seqs = read_fasta(fasta)
    if length_min is not None:
        seqs = [(n, s) for n, s in seqs if len(s) >= length_min]
    if length_max is not None:
        seqs = [(n, s) for n, s in seqs if len(s) <= length_max]
    rows = []
    for name, s in seqs:
        row = {"id": name, "length": len(s)}
        for col, fn in PROXY_FNS.items():
            row[col] = fn(s)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Tier 2: TemStaPro driver
# ----------------------------------------------------------------------

def run_temstapro(fasta: Path,
                  temstapro_dir: Path,
                  emb_dir: Path,
                  out_tsv: Path,
                  python_exe: Path) -> pd.DataFrame:
    """Drive the TemStaPro CLI as a subprocess and load its mean-output TSV.

    Returns DataFrame indexed by sequence id with TemStaPro columns.
    Will download ProtT5 (~1.5 GB) on first run via HuggingFace; will
    fail loudly if `sentencepiece` is not installed in `python_exe`'s env.
    """
    cli = temstapro_dir / "temstapro"
    if not cli.exists():
        raise FileNotFoundError(
            f"TemStaPro CLI not found at {cli}. Setup:\n"
            f"  git clone https://github.com/ievapudz/TemStaPro.git {temstapro_dir}\n"
            f"  {python_exe} -m pip install sentencepiece"
        )
    emb_dir.mkdir(parents=True, exist_ok=True)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python_exe), str(cli),
        "-f", str(fasta),
        "-e", str(emb_dir),
        "-t", str(temstapro_dir),
        "--more-thresholds",
        "--mean-output", str(out_tsv),
    ]
    print(f"[temstapro] running on {fasta.name}")
    print(f"[temstapro] cmd: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"TemStaPro failed (rc={proc.returncode}).\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    if not out_tsv.exists():
        raise RuntimeError(
            f"TemStaPro returned rc=0 but produced no output at {out_tsv}.\n"
            f"This usually means the run silently fell back to CPU/printed to "
            f"stdout instead of writing the TSV — most often because no GPU "
            f"is visible. Check `nvidia-smi` on the executing node.\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    df = pd.read_csv(out_tsv, sep="\t")
    if "sequence_id" in df.columns:
        df = df.rename(columns={"sequence_id": "id"})
    elif df.columns[0].lower() in ("name", "protein", "id"):
        df = df.rename(columns={df.columns[0]: "id"})
    return df


# ----------------------------------------------------------------------
# Distribution comparison
# ----------------------------------------------------------------------

def compare_columns(ref: pd.DataFrame,
                    gen: pd.DataFrame,
                    cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        r = ref[c].dropna().to_numpy()
        g = gen[c].dropna().to_numpy()
        if r.size == 0 or g.size == 0:
            continue
        pooled = np.sqrt(((r.size - 1) * np.var(r, ddof=1)
                          + (g.size - 1) * np.var(g, ddof=1))
                         / (r.size + g.size - 2))
        ks = stats.ks_2samp(r, g, alternative="two-sided", mode="auto")
        rows.append({
            "metric":      c,
            "ref_n":       r.size,
            "gen_n":       g.size,
            "ref_mean":    float(np.mean(r)),
            "gen_mean":    float(np.mean(g)),
            "ref_sd":      float(np.std(r, ddof=1)),
            "gen_sd":      float(np.std(g, ddof=1)),
            "mean_diff":   float(np.mean(g) - np.mean(r)),
            "cohens_d":    float((np.mean(g) - np.mean(r)) / pooled) if pooled > 0 else float("nan"),
            "ks_d":        float(ks.statistic),
            "ks_p":        float(ks.pvalue),
            "wasserstein": float(stats.wasserstein_distance(r, g)),
        })
    return pd.DataFrame(rows)


def plot_distributions(ref: pd.DataFrame,
                       gen: pd.DataFrame,
                       cols: list[str],
                       out_path: Path,
                       ref_label: str,
                       gen_label: str) -> None:
    n = len(cols)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.6 * ncols, 3.0 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for ax, c in zip(axes, cols):
        r = ref[c].dropna().to_numpy()
        g = gen[c].dropna().to_numpy()
        if r.size == 0 and g.size == 0:
            ax.set_visible(False)
            continue
        lo = min(r.min() if r.size else np.inf, g.min() if g.size else np.inf)
        hi = max(r.max() if r.size else -np.inf, g.max() if g.size else -np.inf)
        bins = np.linspace(lo, hi, 50) if hi > lo else 10
        if r.size:
            ax.hist(r, bins=bins, density=True, alpha=0.45,
                    color="#1f77b4", label=ref_label)
        if g.size:
            ax.hist(g, bins=bins, density=True, alpha=0.45,
                    color="#d62728", label=gen_label)
        for x, color in [(r, "#1f77b4"), (g, "#d62728")]:
            if x.size >= 5 and not np.allclose(x, x[0]):
                try:
                    kde = stats.gaussian_kde(x)
                    xs = np.linspace(lo, hi, 256)
                    ax.plot(xs, kde(xs), color=color, lw=1.4)
                except Exception:
                    pass
        ax.set_title(c, fontsize=10)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{gen_label}  vs  {ref_label}  —  thermal-stability proxies", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=Path,
                    default=Path("results/generated_stratified_300_800/sequences.fasta"))
    ap.add_argument("--ref", type=Path,
                    default=Path("pdb_cluster_all_seqs.fasta"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/thermal_stability/stratified_vs_pdb"))
    ap.add_argument("--gen-label", default="generated")
    ap.add_argument("--ref-label", default="PDB (300-800)")
    ap.add_argument("--length-min", type=int, default=300)
    ap.add_argument("--length-max", type=int, default=800)
    # TemStaPro
    ap.add_argument("--temstapro-dir", type=Path, default=None,
                    help="Path to a TemStaPro repo clone. If unset, only Tier 1 runs.")
    ap.add_argument("--temstapro-emb-dir", type=Path, default=Path("/tmp/temstapro_embeddings"),
                    help="Cache dir for ProtT5 embeddings (re-used across runs).")
    ap.add_argument("--python-exe", type=Path, default=Path(sys.executable),
                    help="Python executable for the TemStaPro subprocess.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # --- Tier 1 ---
    print("[tier1] computing sequence proxies...")
    gen_df = tier1_table(args.gen, args.length_min, args.length_max)
    ref_df = tier1_table(args.ref, args.length_min, args.length_max)
    print(f"[tier1] gen n={len(gen_df)}  ref n={len(ref_df)}")

    cols = list(PROXY_FNS.keys())

    # --- Tier 2 (optional) ---
    if args.temstapro_dir is not None:
        print(f"[tier2] running TemStaPro from {args.temstapro_dir}")
        gen_tsp = run_temstapro(args.gen, args.temstapro_dir,
                                args.temstapro_emb_dir / "gen",
                                args.out / "temstapro_gen.tsv",
                                args.python_exe)
        ref_tsp = run_temstapro(args.ref, args.temstapro_dir,
                                args.temstapro_emb_dir / "ref",
                                args.out / "temstapro_ref.tsv",
                                args.python_exe)
        # Numeric columns in TemStaPro's mean output: clf_<thr> probabilities,
        # plus 'left_<thr>' / 'right_<thr>' for upper/lower probability bounds.
        # We keep all numeric columns for comparison.
        tsp_cols = [c for c in gen_tsp.columns
                    if c != "id" and pd.api.types.is_numeric_dtype(gen_tsp[c])]
        gen_df = gen_df.merge(gen_tsp[["id"] + tsp_cols], on="id", how="left")
        ref_df = ref_df.merge(ref_tsp[["id"] + tsp_cols], on="id", how="left")
        cols = cols + tsp_cols
        print(f"[tier2] merged TemStaPro columns: {tsp_cols}")
    else:
        print("[tier2] skipped (no --temstapro-dir given)")

    gen_df.to_csv(args.out / "gen_per_protein.csv", index=False)
    ref_df.to_csv(args.out / "ref_per_protein.csv", index=False)

    summary = compare_columns(ref_df, gen_df, cols)
    summary.to_csv(args.out / "summary.csv", index=False)

    short = summary[["metric", "ref_n", "gen_n",
                     "ref_mean", "gen_mean", "cohens_d",
                     "ref_sd", "gen_sd", "ks_d", "wasserstein"]].copy()
    fmt = {"ref_mean": "{:.3g}", "gen_mean": "{:.3g}", "cohens_d": "{:+.2f}",
           "ref_sd":   "{:.3g}", "gen_sd":   "{:.3g}",
           "ks_d":     "{:.3f}", "wasserstein": "{:.3g}"}
    for col, f in fmt.items():
        short[col] = short[col].map(lambda v, f=f: f.format(v) if pd.notna(v) else "—")
    print()
    print(short.to_string(index=False))
    print()
    print("Legend:")
    print("  cohens_d    : (gen_mean − ref_mean) / pooled_sd")
    print("  ks_d        : 2-sample KS statistic (larger = more different)")
    print("  wasserstein : earth-mover distance, in metric units")

    plot_distributions(ref_df, gen_df, cols,
                       args.out / "distributions.png",
                       args.ref_label, args.gen_label)
    print(f"\n[write] {args.out / 'summary.csv'}")
    print(f"[write] {args.out / 'distributions.png'}")
    print(f"[write] {args.out / 'gen_per_protein.csv'}")
    print(f"[write] {args.out / 'ref_per_protein.csv'}")


if __name__ == "__main__":
    main()
