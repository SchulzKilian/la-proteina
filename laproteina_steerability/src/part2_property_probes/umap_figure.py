"""UMAP visualization: 3x3 grid of property-colored embeddings."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from src.common.io import save_figure
from src.common.plotting import CMAP_SEQUENTIAL
from src.data.loader import ProteinRecord

logger = logging.getLogger(__name__)


def pool_for_umap(
    records: list[ProteinRecord],
    mode: str = "extended",
) -> tuple[np.ndarray, list[str]]:
    """Pool per-residue latents to per-protein vectors for UMAP.

    Parameters
    ----------
    records : list[ProteinRecord]
        Protein records with latents.
    mode : str
        "mean" for ``[mean]`` (8D) or "extended" for
        ``[mean, std, max, min]`` (32D).

    Returns
    -------
    pooled : np.ndarray
        Shape ``[N_proteins, D_pooled]``.
    protein_ids : list[str]
        Protein IDs in order.
    """
    vectors = []
    ids = []
    for rec in records:
        lat = rec.latents
        parts = [lat.mean(axis=0)]
        if mode == "extended":
            parts.extend([
                lat.std(axis=0),
                lat.max(axis=0),
                lat.min(axis=0),
            ])
        vectors.append(np.concatenate(parts))
        ids.append(rec.protein_id)
    return np.array(vectors, dtype=np.float32), ids


def run_umap(
    pooled: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    seed: int = 42,
) -> np.ndarray:
    """Run UMAP on pooled latent vectors.

    Parameters
    ----------
    pooled : np.ndarray
        Shape ``[N, D]``, per-protein pooled latents.
    n_neighbors : int
        UMAP n_neighbors parameter.
    min_dist : float
        UMAP min_dist parameter.
    metric : str
        Distance metric.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape ``[N, 2]``, 2D UMAP embedding.
    """
    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        n_components=2,
    )
    embedding = reducer.fit_transform(pooled)
    return embedding


def plot_umap_grid(
    embedding: np.ndarray,
    protein_ids: list[str],
    prop_df: pd.DataFrame,
    property_names: list[str],
    probe_r2: dict[str, float],
    percentile_clip: list[int] | None = None,
    figures_dir: str = "outputs/figures",
    dpi: int = 300,
    formats: list[str] | None = None,
) -> plt.Figure:
    """Create 3x3 UMAP grid colored by properties.

    Parameters
    ----------
    embedding : np.ndarray
        Shape ``[N, 2]``, UMAP coordinates.
    protein_ids : list[str]
        Protein IDs matching embedding rows.
    prop_df : pd.DataFrame
        Property table with protein_id column.
    property_names : list[str]
        Up to 9 property names to plot.
    probe_r2 : dict[str, float]
        Maps property name to best probe R² for annotation.
    percentile_clip : list[int]
        [low, high] percentiles for color clipping.
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
    if percentile_clip is None:
        percentile_clip = [1, 99]

    # Mean-pool properties to protein level
    prop_protein = prop_df.groupby("protein_id").mean(numeric_only=True).reset_index()
    prop_map = {row["protein_id"]: row for _, row in prop_protein.iterrows()}

    n_props = min(len(property_names), 9)
    nrows = 3
    ncols = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 12))
    axes = axes.flatten()

    for i in range(n_props):
        ax = axes[i]
        pname = property_names[i]

        # Get values aligned to embedding order
        vals = []
        for pid in protein_ids:
            if pid in prop_map and pname in prop_map[pid].index:
                vals.append(float(prop_map[pid][pname]))
            else:
                vals.append(float("nan"))
        vals = np.array(vals)

        # Mask NaN
        valid = ~np.isnan(vals)
        if valid.sum() < 2:
            ax.text(0.5, 0.5, f"{pname}\n(no data)", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(pname)
            continue

        # Clip to percentiles
        vmin = np.nanpercentile(vals[valid], percentile_clip[0])
        vmax = np.nanpercentile(vals[valid], percentile_clip[1])

        scatter = ax.scatter(
            embedding[valid, 0], embedding[valid, 1],
            c=np.clip(vals[valid], vmin, vmax),
            cmap=CMAP_SEQUENTIAL,
            s=8, alpha=0.7, edgecolors="none",
            norm=Normalize(vmin=vmin, vmax=vmax),
        )
        plt.colorbar(scatter, ax=ax, shrink=0.8)

        r2_str = f"R²={probe_r2.get(pname, float('nan')):.2f}"
        ax.set_title(f"{pname} ({r2_str})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused panels
    for i in range(n_props, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "UMAP of pooled per-protein latents, colored by property\n"
        "(Probes operate on unpooled per-residue latents; this figure illustrates rather than tests)",
        fontsize=11, y=1.01,
    )

    save_figure(fig, "umap_property_grid", figures_dir, dpi=dpi, formats=formats)
    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data.loader import make_synthetic_dataset

    records = make_synthetic_dataset(n_proteins=30, length=50)
    pooled, ids = pool_for_umap(records, mode="extended")
    print(f"Pooled shape: {pooled.shape}")

    embedding = run_umap(pooled, seed=42)
    print(f"Embedding shape: {embedding.shape}")
    plt.close("all")
