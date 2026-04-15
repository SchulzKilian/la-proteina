"""PCA, effective rank, utilization, and length-sensitivity analyses."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.io import save_figure, save_table
from src.data.loader import ProteinRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PCA and effective rank
# ---------------------------------------------------------------------------

def compute_pca(latents: np.ndarray) -> dict:
    """Run PCA on pooled latents (centered, not scaled).

    Parameters
    ----------
    latents : np.ndarray
        Shape ``[N, D]``, pooled residue latents.

    Returns
    -------
    dict
        Keys: eigenvalues, explained_variance_ratio, cumulative_variance,
        participation_ratio, mean (the centering vector), components.
    """
    mean = latents.mean(axis=0)
    centered = latents - mean
    cov = np.cov(centered, rowvar=False)
    eigenvalues, components = np.linalg.eigh(cov)

    # Sort descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = components[:, order]

    total_var = eigenvalues.sum()
    explained_ratio = eigenvalues / total_var
    cumulative = np.cumsum(explained_ratio)

    # Participation ratio: (sum lambda)^2 / sum(lambda^2)
    participation_ratio = float(total_var ** 2 / (eigenvalues ** 2).sum())

    return {
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained_ratio,
        "cumulative_variance": cumulative,
        "participation_ratio": participation_ratio,
        "mean": mean,
        "components": components,
    }


def effective_rank_at_thresholds(
    cumulative_variance: np.ndarray,
    thresholds: list[float],
) -> dict[float, int]:
    """Compute effective rank at given explained-variance thresholds.

    Parameters
    ----------
    cumulative_variance : np.ndarray
        Cumulative explained variance ratio (ascending).
    thresholds : list[float]
        E.g. [0.90, 0.95, 0.99].

    Returns
    -------
    dict[float, int]
        Threshold -> number of components needed.
    """
    ranks = {}
    for t in thresholds:
        idx = np.searchsorted(cumulative_variance, t)
        ranks[t] = int(idx + 1)  # 1-indexed
    return ranks


def pca_results_table(pca: dict, thresholds: list[float]) -> pd.DataFrame:
    """Build a summary table of PCA results.

    Parameters
    ----------
    pca : dict
        Output of ``compute_pca``.
    thresholds : list[float]
        Explained variance thresholds for effective rank.

    Returns
    -------
    pd.DataFrame
    """
    D = len(pca["eigenvalues"])
    ranks = effective_rank_at_thresholds(pca["cumulative_variance"], thresholds)

    rows = []
    for i in range(D):
        rows.append({
            "component": i,
            "eigenvalue": pca["eigenvalues"][i],
            "explained_variance_ratio": pca["explained_variance_ratio"][i],
            "cumulative_variance": pca["cumulative_variance"][i],
        })
    df = pd.DataFrame(rows)
    # Add summary rows
    df.attrs["participation_ratio"] = pca["participation_ratio"]
    df.attrs["effective_ranks"] = ranks
    return df


def plot_pca(
    pca: dict,
    thresholds: list[float],
    figures_dir: str = "outputs/figures",
    dpi: int = 300,
    formats: list[str] | None = None,
) -> plt.Figure:
    """Scree plot (linear + log) and cumulative explained variance.

    Parameters
    ----------
    pca : dict
        Output of ``compute_pca``.
    thresholds : list[float]
        Thresholds to annotate on cumulative plot.
    figures_dir : str
        Output directory.
    dpi : int
        Figure DPI.
    formats : list[str]
        File formats.

    Returns
    -------
    plt.Figure
    """
    eigvals = pca["eigenvalues"]
    expl = pca["explained_variance_ratio"]
    cumul = pca["cumulative_variance"]
    pr = pca["participation_ratio"]
    D = len(eigvals)
    x = np.arange(D)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    # Scree (linear)
    ax1.bar(x, eigvals, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_title("Scree plot (linear)")
    ax1.set_xticks(x)

    # Scree (log)
    ax2.bar(x, eigvals, color="steelblue", alpha=0.8)
    ax2.set_yscale("log")
    ax2.set_xlabel("Component")
    ax2.set_ylabel("Eigenvalue (log)")
    ax2.set_title("Scree plot (log)")
    ax2.set_xticks(x)

    # Cumulative explained variance
    ax3.plot(x, cumul, "o-", color="darkred", markersize=5)
    ax3.set_xlabel("Component")
    ax3.set_ylabel("Cumulative explained variance")
    ax3.set_title("Cumulative variance")
    ax3.set_xticks(x)
    ax3.set_ylim(0, 1.05)

    # Annotate thresholds — stagger vertically to avoid overlap
    ranks = effective_rank_at_thresholds(cumul, thresholds)
    for i, (t, r) in enumerate(sorted(ranks.items())):
        ax3.axhline(t, ls="--", color="gray", alpha=0.4)
        # Place label at left side, offset down to avoid stacking
        ax3.text(0.05, t - 0.03, f"{t:.0%} @ {r}D",
                 ha="left", fontsize=8, color="dimgray")

    # Annotate participation ratio
    ax3.text(0.02, 0.02, f"Participation ratio: {pr:.2f} / {D}",
             transform=ax3.transAxes, fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    save_figure(fig, "pca_analysis", figures_dir, dpi=dpi, formats=formats)
    return fig


# ---------------------------------------------------------------------------
# Per-dimension utilization
# ---------------------------------------------------------------------------

def compute_utilization(records: list[ProteinRecord]) -> pd.DataFrame:
    """Compute between-protein vs within-protein variance per dimension.

    For each latent dimension, computes:
    - total variance across all residues
    - mean within-protein variance (averaged across proteins)
    - between-protein variance (of protein means)
    - within/total ratio

    A high within/total ratio means the dimension varies mostly within
    proteins (per-residue property). A low ratio means it varies between
    proteins (global property).

    Parameters
    ----------
    records : list[ProteinRecord]
        Loaded protein records.

    Returns
    -------
    pd.DataFrame
        One row per dimension.
    """
    all_latents = np.concatenate([r.latents for r in records], axis=0)
    D = all_latents.shape[1]
    total_var = all_latents.var(axis=0)

    # Within-protein variance: average of per-protein variances
    within_vars = []
    protein_means = []
    for rec in records:
        if rec.length > 1:
            within_vars.append(rec.latents.var(axis=0))
        else:
            within_vars.append(np.zeros(D))
        protein_means.append(rec.latents.mean(axis=0))

    mean_within = np.mean(within_vars, axis=0)
    between_var = np.var(protein_means, axis=0)

    rows = []
    for d in range(D):
        rows.append({
            "dim": d,
            "total_variance": float(total_var[d]),
            "mean_within_variance": float(mean_within[d]),
            "between_variance": float(between_var[d]),
            "within_total_ratio": float(mean_within[d] / total_var[d]) if total_var[d] > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def plot_utilization(
    util_df: pd.DataFrame,
    figures_dir: str = "outputs/figures",
    dpi: int = 300,
    formats: list[str] | None = None,
) -> plt.Figure:
    """Bar chart of within vs between protein variance per dimension.

    Parameters
    ----------
    util_df : pd.DataFrame
        Output of ``compute_utilization``.
    figures_dir : str
        Output directory.
    dpi : int
        Figure DPI.
    formats : list[str]
        File formats.

    Returns
    -------
    plt.Figure
    """
    D = len(util_df)
    x = np.arange(D)
    width = 0.35
    labels = [f"D{d}" for d in range(D)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    # Left: stacked bar (within + between = total)
    ax1.bar(x, util_df["mean_within_variance"],
            label="Within-protein (mean)", color="steelblue", alpha=0.8)
    ax1.bar(x, util_df["between_variance"],
            bottom=util_df["mean_within_variance"],
            label="Between-protein", color="coral", alpha=0.8)
    ax1.set_xlabel("Latent dimension")
    ax1.set_ylabel("Variance")
    ax1.set_title("Variance decomposition (stacked)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()

    # Right: within/total ratio — the key diagnostic
    ratios = util_df["within_total_ratio"]
    colors = ["steelblue" if r > 0.5 else "coral" for r in ratios]
    ax2.bar(x, ratios, color=colors, alpha=0.8)
    ax2.axhline(0.5, ls="--", color="gray", alpha=0.5, label="50% threshold")
    ax2.set_xlabel("Latent dimension")
    ax2.set_ylabel("Within / Total variance ratio")
    ax2.set_title("Per-dim utilization ratio (high = per-residue, low = global)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.05)
    # Annotate each bar
    for i, r in enumerate(ratios):
        ax2.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
    ax2.legend()

    save_figure(fig, "dim_utilization", figures_dir, dpi=dpi, formats=formats)
    return fig


# ---------------------------------------------------------------------------
# Length sensitivity
# ---------------------------------------------------------------------------

def compute_length_stats(records: list[ProteinRecord]) -> pd.DataFrame:
    """Compute per-protein latent statistics vs length.

    Parameters
    ----------
    records : list[ProteinRecord]
        Loaded protein records.

    Returns
    -------
    pd.DataFrame
        One row per protein with columns: protein_id, length,
        mean_l2_norm, dim_0_mean, ..., dim_D_mean.
    """
    rows = []
    D = records[0].latents.shape[1]
    for rec in records:
        norms = np.linalg.norm(rec.latents, axis=1)
        row = {
            "protein_id": rec.protein_id,
            "length": rec.length,
            "mean_l2_norm": float(norms.mean()),
        }
        means = rec.latents.mean(axis=0)
        for d in range(D):
            row[f"dim_{d}_mean"] = float(means[d])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_length_sensitivity(
    length_df: pd.DataFrame,
    figures_dir: str = "outputs/figures",
    dpi: int = 300,
    formats: list[str] | None = None,
) -> plt.Figure:
    """Plot latent norm and per-dim means vs protein length.

    Parameters
    ----------
    length_df : pd.DataFrame
        Output of ``compute_length_stats``.
    figures_dir : str
        Output directory.
    dpi : int
        Figure DPI.
    formats : list[str]
        File formats.

    Returns
    -------
    plt.Figure
    """
    dim_cols = [c for c in length_df.columns if c.startswith("dim_")]
    n_dims = len(dim_cols)
    N = len(length_df)

    # Adaptive alpha: more transparent at large N, opaque at small N
    alpha = max(0.15, min(0.8, 50.0 / N))
    n_bins = max(5, min(20, N // 5))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: mean L2 norm vs length
    ax = axes[0]
    ax.scatter(length_df["length"], length_df["mean_l2_norm"],
               alpha=alpha, s=max(10, 80 / (N ** 0.3)), color="steelblue",
               edgecolors="none", zorder=2)
    # Binned mean
    bins = pd.cut(length_df["length"], bins=n_bins)
    binned = length_df.groupby(bins, observed=True)["mean_l2_norm"].mean()
    bin_centers = [interval.mid for interval in binned.index]
    ax.plot(bin_centers, binned.values, "o-", color="darkred",
            markersize=5, lw=2, zorder=3, label="Binned mean")
    ax.set_xlabel("Protein length (residues)")
    ax.set_ylabel("Mean per-residue L2 norm")
    ax.set_title("Latent norm vs protein length")
    ax.legend()

    # Compute Pearson r for annotation
    from scipy.stats import pearsonr
    r, p = pearsonr(length_df["length"], length_df["mean_l2_norm"])
    ax.text(0.02, 0.98, f"r = {r:.3f} (p = {p:.2e})",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Right: per-dim mean vs length, one color per dim with Pearson r
    ax = axes[1]
    cmap = plt.get_cmap("tab10")
    for i, col in enumerate(dim_cols):
        dim_label = col.replace("dim_", "D").replace("_mean", "")
        r_val, _ = pearsonr(length_df["length"], length_df[col])
        ax.scatter(length_df["length"], length_df[col],
                   alpha=alpha, s=max(10, 80 / (N ** 0.3)),
                   color=cmap(i), edgecolors="none",
                   label=f"{dim_label} (r={r_val:.2f})")
    ax.set_xlabel("Protein length (residues)")
    ax.set_ylabel("Per-protein mean of latent dim")
    ax.set_title("Per-dim mean vs protein length")
    ax.legend(fontsize=7, ncol=2, loc="best",
              framealpha=0.8, edgecolor="gray")

    save_figure(fig, "length_sensitivity", figures_dir, dpi=dpi, formats=formats)
    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data.loader import make_synthetic_dataset, pool_latents

    records = make_synthetic_dataset(n_proteins=10, length=50)
    latents, _ = pool_latents(records)

    pca = compute_pca(latents)
    print(f"Participation ratio: {pca['participation_ratio']:.2f}")
    print(f"Eigenvalues: {pca['eigenvalues']}")

    util_df = compute_utilization(records)
    print("\nUtilization:")
    print(util_df.to_string())

    length_df = compute_length_stats(records)
    print(f"\nLength stats: {len(length_df)} proteins")
    plt.close("all")
