"""Marginal distribution analysis and pairwise structure for latent dimensions."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

from src.common.io import save_figure, save_table
from src.common.plotting import CMAP_CORRELATION

logger = logging.getLogger(__name__)


def compute_marginal_stats(
    latents: np.ndarray,
    shapiro_subsample: int = 5000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Compute per-dimension marginal statistics.

    Parameters
    ----------
    latents : np.ndarray
        Shape ``[N, D]``, pooled residue latents.
    shapiro_subsample : int
        Number of residues to subsample for Shapiro-Wilk test
        (full dataset is too large).
    rng : np.random.Generator
        For reproducible subsampling.

    Returns
    -------
    pd.DataFrame
        One row per latent dimension with columns: dim, mean, std,
        skewness, kurtosis, min, max, p1, p99, shapiro_W, shapiro_p,
        shapiro_n.
    """
    N, D = latents.shape
    if rng is None:
        rng = np.random.default_rng(0)

    sub_idx = rng.choice(N, size=min(shapiro_subsample, N), replace=False)

    rows = []
    for d in range(D):
        col = latents[:, d]
        sub = col[sub_idx]
        W, p = stats.shapiro(sub)
        rows.append({
            "dim": d,
            "mean": float(col.mean()),
            "std": float(col.std()),
            "skewness": float(stats.skew(col)),
            "kurtosis": float(stats.kurtosis(col)),
            "min": float(col.min()),
            "max": float(col.max()),
            "p1": float(np.percentile(col, 1)),
            "p99": float(np.percentile(col, 99)),
            "shapiro_W": float(W),
            "shapiro_p": float(p),
            "shapiro_n": len(sub),
        })
    return pd.DataFrame(rows)


def plot_marginals(
    latents: np.ndarray,
    n_bins: int = 50,
    figures_dir: str = "outputs/figures",
    dpi: int = 300,
    formats: list[str] | None = None,
) -> plt.Figure:
    """Plot histogram + KDE for each latent dimension in a 4x2 grid.

    Parameters
    ----------
    latents : np.ndarray
        Shape ``[N, D]``, pooled residue latents.
    n_bins : int
        Number of histogram bins.
    figures_dir : str
        Output directory for saved figure.
    dpi : int
        Figure DPI.
    formats : list[str]
        File formats to save.

    Returns
    -------
    plt.Figure
    """
    D = latents.shape[1]
    nrows, ncols = (D + 1) // 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3 * nrows))
    axes = axes.flatten()

    for d in range(D):
        ax = axes[d]
        ax.hist(latents[:, d], bins=n_bins, density=True, alpha=0.6,
                color="steelblue", edgecolor="none")
        # KDE overlay
        xmin, xmax = latents[:, d].min(), latents[:, d].max()
        margin = (xmax - xmin) * 0.05
        xs = np.linspace(xmin - margin, xmax + margin, 200)
        kde = stats.gaussian_kde(latents[:, d])
        ax.plot(xs, kde(xs), color="darkred", lw=1.5)
        ax.set_title(f"Dim {d}")
        ax.set_ylabel("Density")

    # Hide unused axes
    for i in range(D, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Latent marginal distributions", fontsize=14, y=1.02)
    save_figure(fig, "latent_marginals", figures_dir, dpi=dpi, formats=formats)
    return fig


def compute_correlation_matrices(
    latents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Pearson and Spearman correlation matrices.

    Parameters
    ----------
    latents : np.ndarray
        Shape ``[N, D]``, pooled residue latents.

    Returns
    -------
    pearson : np.ndarray
        Shape ``[D, D]``.
    spearman : np.ndarray
        Shape ``[D, D]``.
    """
    pearson = np.corrcoef(latents, rowvar=False)
    spearman, _ = stats.spearmanr(latents)
    if latents.shape[1] == 1:
        spearman = np.array([[1.0]])
    return pearson, spearman


def compute_mutual_information(
    latents: np.ndarray,
    subsample: int = 10000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compute mutual information matrix between latent dimensions.

    Uses sklearn's ``mutual_info_regression`` on a subsample. MI is
    estimated pairwise: for each (i, j) pair, MI(X_i, X_j) is computed
    by treating X_j as the target.

    Parameters
    ----------
    latents : np.ndarray
        Shape ``[N, D]``, pooled residue latents.
    subsample : int
        Number of residues to subsample for MI estimation.
    rng : np.random.Generator
        For reproducible subsampling.

    Returns
    -------
    np.ndarray
        Shape ``[D, D]`` MI matrix (symmetric, diagonal is self-MI).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N, D = latents.shape
    idx = rng.choice(N, size=min(subsample, N), replace=False)
    sub = latents[idx]

    mi_matrix = np.zeros((D, D))
    seed = int(rng.integers(0, 2**31))
    for j in range(D):
        mi_values = mutual_info_regression(
            sub, sub[:, j], random_state=seed,
        )
        mi_matrix[:, j] = mi_values

    # Symmetrize
    mi_matrix = (mi_matrix + mi_matrix.T) / 2
    return mi_matrix


def plot_correlation_heatmaps(
    pearson: np.ndarray,
    spearman: np.ndarray,
    figures_dir: str = "outputs/figures",
    dpi: int = 300,
    formats: list[str] | None = None,
) -> plt.Figure:
    """Plot Pearson and Spearman correlation matrices side by side.

    Parameters
    ----------
    pearson, spearman : np.ndarray
        Correlation matrices, shape ``[D, D]``.
    figures_dir : str
        Output directory.
    dpi : int
        Figure DPI.
    formats : list[str]
        File formats to save.

    Returns
    -------
    plt.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    D = pearson.shape[0]
    labels = [f"D{i}" for i in range(D)]

    sns.heatmap(pearson, ax=ax1, vmin=-1, vmax=1, cmap=CMAP_CORRELATION,
                annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels,
                square=True)
    ax1.set_title("Pearson correlation")

    sns.heatmap(spearman, ax=ax2, vmin=-1, vmax=1, cmap=CMAP_CORRELATION,
                annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels,
                square=True)
    ax2.set_title("Spearman correlation")

    save_figure(fig, "latent_correlations", figures_dir, dpi=dpi, formats=formats)
    return fig


def plot_mi_heatmap(
    mi_matrix: np.ndarray,
    figures_dir: str = "outputs/figures",
    dpi: int = 300,
    formats: list[str] | None = None,
) -> plt.Figure:
    """Plot mutual information matrix as a heatmap.

    Parameters
    ----------
    mi_matrix : np.ndarray
        Shape ``[D, D]``.
    figures_dir : str
        Output directory.
    dpi : int
        Figure DPI.
    formats : list[str]
        File formats to save.

    Returns
    -------
    plt.Figure
    """
    D = mi_matrix.shape[0]
    labels = [f"D{i}" for i in range(D)]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mi_matrix, ax=ax, cmap="YlOrRd", annot=True, fmt=".3f",
                xticklabels=labels, yticklabels=labels, square=True)
    ax.set_title("Mutual information (nats)")
    save_figure(fig, "latent_mutual_information", figures_dir, dpi=dpi, formats=formats)
    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data.loader import make_synthetic_dataset, pool_latents

    records = make_synthetic_dataset(n_proteins=10, length=50)
    latents, _ = pool_latents(records)
    rng = np.random.default_rng(42)

    stats_df = compute_marginal_stats(latents, shapiro_subsample=200, rng=rng)
    print(stats_df.to_string())

    pearson, spearman = compute_correlation_matrices(latents)
    print(f"\nMax off-diag Pearson: {np.abs(pearson - np.eye(pearson.shape[0])).max():.3f}")

    mi = compute_mutual_information(latents, subsample=200, rng=rng)
    print(f"Max off-diag MI: {(mi - np.diag(np.diag(mi))).max():.3f}")
    plt.close("all")
