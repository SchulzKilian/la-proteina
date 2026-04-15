"""Property-property correlation analysis and clustering."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from src.common.io import save_figure, save_table
from src.common.plotting import CMAP_CORRELATION

logger = logging.getLogger(__name__)


def compute_property_correlations(
    prop_df: pd.DataFrame,
    property_names: list[str],
    property_granularity: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson and Spearman correlations between properties.

    For per-residue properties, mean-pools to protein level first.

    Parameters
    ----------
    prop_df : pd.DataFrame
        Property table with protein_id and property columns.
    property_names : list[str]
        Columns to correlate.
    property_granularity : dict[str, str]
        Maps property name to "protein" or "residue".

    Returns
    -------
    pearson : pd.DataFrame
        Pearson correlation matrix.
    spearman : pd.DataFrame
        Spearman correlation matrix.
    """
    # Mean-pool all properties to protein level
    available = [p for p in property_names if p in prop_df.columns]
    agg_dict = {}
    for p in available:
        agg_dict[p] = "mean"

    protein_level = prop_df.groupby("protein_id").agg(agg_dict).reset_index()

    # Drop proteins with any NaN
    data = protein_level[available].dropna()

    if len(data) < 3:
        logger.warning("Too few valid proteins for correlation: %d", len(data))
        empty = pd.DataFrame(np.nan, index=available, columns=available)
        return empty, empty.copy()

    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")
    return pearson, spearman


def cluster_properties(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.8,
) -> pd.DataFrame:
    """Hierarchical clustering of properties by correlation.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix (properties x properties).
    threshold : float
        Correlation threshold for cluster membership. Properties with
        |corr| > threshold are in the same cluster.

    Returns
    -------
    pd.DataFrame
        Columns: property, cluster_id.
    """
    props = corr_matrix.columns.tolist()
    n = len(props)

    if n < 2:
        return pd.DataFrame({"property": props, "cluster_id": range(n)})

    # Distance = 1 - |corr|
    dist_matrix = 1.0 - np.abs(corr_matrix.values)
    np.fill_diagonal(dist_matrix, 0)
    # Ensure symmetry and non-negative
    dist_matrix = np.clip((dist_matrix + dist_matrix.T) / 2, 0, None)

    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="average")

    # Cut at distance = 1 - threshold
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    return pd.DataFrame({"property": props, "cluster_id": labels.tolist()})


def find_goodhart_pairs(
    corr_matrix: pd.DataFrame,
    pairs: list[list[str]],
    divergence_threshold: float = 0.5,
) -> pd.DataFrame:
    """Identify Goodhart pairs: conceptually similar but empirically divergent.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix (Pearson or Spearman).
    pairs : list[list[str]]
        Pairs of properties to check, e.g. [["camsol", "swi"]].
    divergence_threshold : float
        If |corr| < this threshold, the pair diverges.

    Returns
    -------
    pd.DataFrame
        Columns: prop_a, prop_b, correlation, diverges.
    """
    rows = []
    for pair in pairs:
        a, b = pair[0], pair[1]
        if a in corr_matrix.columns and b in corr_matrix.columns:
            corr_val = float(corr_matrix.loc[a, b])
            rows.append({
                "prop_a": a,
                "prop_b": b,
                "correlation": corr_val,
                "diverges": abs(corr_val) < divergence_threshold,
            })
        else:
            missing = [x for x in [a, b] if x not in corr_matrix.columns]
            logger.warning("Goodhart pair (%s, %s) skipped — missing: %s", a, b, missing)
    return pd.DataFrame(rows)


def plot_clustered_heatmap(
    corr_matrix: pd.DataFrame,
    figures_dir: str = "outputs/figures",
    dpi: int = 300,
    formats: list[str] | None = None,
) -> plt.Figure:
    """Plot a hierarchically clustered correlation heatmap.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix (properties x properties).
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
    # Temporarily disable constrained_layout (incompatible with clustermap)
    import matplotlib as mpl
    old_layout = mpl.rcParams.get("figure.constrained_layout.use", False)
    mpl.rcParams["figure.constrained_layout.use"] = False

    g = sns.clustermap(
        corr_matrix,
        cmap=CMAP_CORRELATION,
        vmin=-1, vmax=1,
        annot=True, fmt=".2f",
        figsize=(8, 7),
        linewidths=0.5,
        method="average",
        metric="correlation",
    )
    mpl.rcParams["figure.constrained_layout.use"] = old_layout
    g.fig.suptitle("Property correlation (clustered)", y=1.02)

    # Save via our utility
    from src.common.io import save_figure
    save_figure(g.fig, "property_correlation_clustered", figures_dir,
                dpi=dpi, formats=formats)
    return g.fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Smoke test with random data
    rng = np.random.default_rng(42)
    props = ["net_charge", "rg", "swi", "tango"]
    n = 50
    data = pd.DataFrame({
        "protein_id": [f"p{i}" for i in range(n)],
        **{p: rng.standard_normal(n) for p in props},
    })

    pearson, spearman = compute_property_correlations(data, props, {p: "protein" for p in props})
    print("Pearson:\n", pearson)

    clusters = cluster_properties(pearson, threshold=0.8)
    print("\nClusters:\n", clusters)

    goodhart = find_goodhart_pairs(pearson, [["swi", "tango"]])
    print("\nGoodhart:\n", goodhart)
    plt.close("all")
