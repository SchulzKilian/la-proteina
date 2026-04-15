"""Part 1 driver: Latent geometry diagnostics.

Run with:
    python -m src.part1_latent_geometry.run --config config/default.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.common.io import save_table, write_run_metadata
from src.common.plotting import apply_style
from src.data.loader import load_dataset, make_synthetic_dataset, pool_latents
from src.part1_latent_geometry.distributions import (
    compute_correlation_matrices,
    compute_marginal_stats,
    compute_mutual_information,
    plot_correlation_heatmaps,
    plot_marginals,
    plot_mi_heatmap,
)
from src.part1_latent_geometry.pca import (
    compute_length_stats,
    compute_pca,
    compute_utilization,
    pca_results_table,
    plot_length_sensitivity,
    plot_pca,
    plot_utilization,
)

logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run(config: dict, use_synthetic: bool = False) -> None:
    """Execute all Part 1 analyses.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.
    use_synthetic : bool
        If True, use synthetic data instead of loading from disk.
    """
    apply_style()

    seed = config["seed"]
    rng = np.random.default_rng(seed)

    fig_dir = config["outputs"]["figures_dir"]
    tab_dir = config["outputs"]["tables_dir"]
    out_dir = config["outputs"]["base_dir"]
    dpi = config["outputs"]["figure_dpi"]
    formats = config["outputs"]["figure_formats"]
    p1 = config["part1"]

    # --- Load data ---
    if use_synthetic:
        logger.info("Using synthetic dataset")
        records = make_synthetic_dataset(n_proteins=10, length=50, rng=rng)
    else:
        data_cfg = config["data"]
        length_range = data_cfg.get("length_range")
        if length_range is not None:
            length_range = tuple(length_range)
        records = load_dataset(
            latent_dir=data_cfg["latent_dir"],
            file_format=data_cfg["file_format"],
            field_names=data_cfg["field_names"],
            load_coords=False,
            subsample=data_cfg.get("subsample"),
            rng=rng,
            length_range=length_range,
        )

    all_latents, all_ids = pool_latents(records)
    n_proteins = len(records)
    n_residues, latent_dim = all_latents.shape
    logger.info("Dataset: %d proteins, %d residues, %d latent dims",
                n_proteins, n_residues, latent_dim)

    # --- Step 2: Marginal distributions ---
    logger.info("Step 2: Marginal distributions")
    marginal_stats = compute_marginal_stats(
        all_latents,
        shapiro_subsample=p1["marginals"]["shapiro_subsample"],
        rng=rng,
    )
    save_table(marginal_stats, "marginal_stats", tab_dir)
    plot_marginals(all_latents, n_bins=p1["marginals"]["n_bins"],
                   figures_dir=fig_dir, dpi=dpi, formats=formats)
    plt.close("all")

    # --- Step 3: Pairwise structure ---
    logger.info("Step 3: Pairwise correlations and MI")
    pearson, spearman = compute_correlation_matrices(all_latents)
    save_table(
        _matrix_to_df(pearson, "pearson"), "pearson_correlation", tab_dir,
        include_index=True,
    )
    save_table(
        _matrix_to_df(spearman, "spearman"), "spearman_correlation", tab_dir,
        include_index=True,
    )
    plot_correlation_heatmaps(pearson, spearman, figures_dir=fig_dir,
                              dpi=dpi, formats=formats)
    plt.close("all")

    mi = compute_mutual_information(
        all_latents,
        subsample=p1["correlations"]["mi_subsample"],
        rng=rng,
    )
    save_table(_matrix_to_df(mi, "MI"), "mutual_information", tab_dir,
               include_index=True)
    plot_mi_heatmap(mi, figures_dir=fig_dir, dpi=dpi, formats=formats)
    plt.close("all")

    # --- Step 4: PCA and effective rank ---
    logger.info("Step 4: PCA analysis")
    pca = compute_pca(all_latents)
    thresholds = p1["pca"]["explained_variance_thresholds"]
    pca_df = pca_results_table(pca, thresholds)
    save_table(pca_df, "pca_results", tab_dir)
    plot_pca(pca, thresholds, figures_dir=fig_dir, dpi=dpi, formats=formats)
    plt.close("all")

    from src.part1_latent_geometry.pca import effective_rank_at_thresholds
    eff_ranks = effective_rank_at_thresholds(pca["cumulative_variance"], thresholds)

    # --- Step 5: Utilization ---
    logger.info("Step 5: Per-dimension utilization")
    util_df = compute_utilization(records)
    save_table(util_df, "dim_utilization", tab_dir)
    plot_utilization(util_df, figures_dir=fig_dir, dpi=dpi, formats=formats)
    plt.close("all")

    # --- Step 6: Length sensitivity ---
    logger.info("Step 6: Length sensitivity")
    length_df = compute_length_stats(records)
    save_table(length_df, "length_stats", tab_dir)
    plot_length_sensitivity(length_df, figures_dir=fig_dir, dpi=dpi, formats=formats)
    plt.close("all")

    # --- Metadata ---
    write_run_metadata(out_dir, config, dataset_size=n_proteins, extra={
        "part": "part1_latent_geometry",
        "n_residues": n_residues,
        "latent_dim": latent_dim,
    })

    # --- Summary ---
    collapse_thresh = p1["utilization"]["collapse_threshold"]
    max_var = util_df["total_variance"].max()
    n_collapsed = int((util_df["total_variance"] < collapse_thresh * max_var).sum())
    max_pearson_offdiag = float(np.abs(
        pearson - np.eye(pearson.shape[0])
    ).max())

    summary = _build_summary(
        n_proteins=n_proteins,
        n_residues=n_residues,
        latent_dim=latent_dim,
        participation_ratio=pca["participation_ratio"],
        eff_ranks=eff_ranks,
        max_pearson_offdiag=max_pearson_offdiag,
        n_collapsed=n_collapsed,
        collapse_thresh=collapse_thresh,
    )
    summary_path = Path(out_dir) / "part1_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary)
    logger.info("Part 1 complete. Summary: %s", summary_path)


def _matrix_to_df(matrix: np.ndarray, name: str) -> "pd.DataFrame":
    """Convert a DxD matrix to a labeled DataFrame."""
    import pandas as pd
    D = matrix.shape[0]
    labels = [f"D{i}" for i in range(D)]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def _build_summary(
    n_proteins: int,
    n_residues: int,
    latent_dim: int,
    participation_ratio: float,
    eff_ranks: dict[float, int],
    max_pearson_offdiag: float,
    n_collapsed: int,
    collapse_thresh: float,
) -> str:
    ranks_str = ", ".join(f"{k:.0%}: {v}D" for k, v in sorted(eff_ranks.items()))
    return f"""# Part 1: Latent Geometry Summary

## Dataset
- Proteins: {n_proteins}
- Total residues: {n_residues}
- Latent dimensions: {latent_dim}

## Key results

| Metric | Value |
|--------|-------|
| Participation ratio | {participation_ratio:.3f} / {latent_dim} |
| Effective rank | {ranks_str} |
| Max off-diagonal Pearson | {max_pearson_offdiag:.3f} |
| Collapsed dims (var < {collapse_thresh:.0%} of max) | {n_collapsed} |

## Interpretation
- Participation ratio close to {latent_dim} means all dimensions contribute.
- No collapsed dimensions = no posterior collapse.
- High off-diagonal correlation (>0.5) suggests redundancy.

## Figures
- `figures/latent_marginals.{{png,pdf}}` — per-dim histograms + KDE
- `figures/latent_correlations.{{png,pdf}}` — Pearson + Spearman heatmaps
- `figures/latent_mutual_information.{{png,pdf}}` — MI heatmap
- `figures/pca_analysis.{{png,pdf}}` — scree + cumulative variance
- `figures/dim_utilization.{{png,pdf}}` — within vs between variance
- `figures/length_sensitivity.{{png,pdf}}` — latent norm vs length

## Tables
- `tables/marginal_stats.csv`
- `tables/pearson_correlation.csv`, `tables/spearman_correlation.csv`
- `tables/mutual_information.csv`
- `tables/pca_results.csv`
- `tables/dim_utilization.csv`
- `tables/length_stats.csv`
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 1: Latent geometry diagnostics")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for smoke testing")
    args = parser.parse_args()

    config = _load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config.get("log_level", "INFO")),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    run(config, use_synthetic=args.synthetic)


if __name__ == "__main__":
    main()
