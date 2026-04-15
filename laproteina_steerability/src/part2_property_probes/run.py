"""Part 2 driver: Property probes and steerability diagnostic.

Run with:
    python -m src.part2_property_probes.run --config config/default.yaml

t-convention (matching La-Proteina codebase):
    t=0 is pure noise, t=1 is clean data.
    z_t = (1-t)*noise + t*z_clean

WARNING: It has NOT been verified whether the steering predictor training
uses this same convention. If probe results look inverted at low vs high t,
the convention may be flipped. All outputs log the convention used.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.common.io import save_table, write_run_metadata
from src.common.plotting import apply_style
from src.data.loader import (
    ProteinRecord,
    load_dataset,
    make_synthetic_dataset,
    pool_latents,
)
from src.data.noising import make_noised_latents
from src.part2_property_probes.properties import (
    align_properties_to_latents,
    load_properties,
    prepare_probe_inputs,
)
from src.part2_property_probes.property_corr import (
    cluster_properties,
    compute_property_correlations,
    find_goodhart_pairs,
    plot_clustered_heatmap,
)
from src.part2_property_probes.probes import run_all_probes
from src.part2_property_probes.umap_figure import (
    plot_umap_grid,
    pool_for_umap,
    run_umap,
)

logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _make_synthetic_properties(
    records: list[ProteinRecord],
    property_names: list[str],
    property_granularity: dict[str, str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create synthetic property data aligned to synthetic latent records.

    Some properties are deliberately correlated with latent features to
    make probes non-trivial in smoke tests.

    Parameters
    ----------
    records : list[ProteinRecord]
        Synthetic protein records.
    property_names : list[str]
        Properties to generate.
    property_granularity : dict[str, str]
        Maps property to "protein" or "residue".
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for rec in records:
        # Create protein-level and residue-level features
        mean_lat = rec.latents.mean(axis=0)

        if any(property_granularity.get(p) == "residue" for p in property_names):
            # Residue-level: one row per residue
            for res_idx in range(rec.length):
                row = {
                    "protein_id": rec.protein_id,
                    "residue_index": res_idx,
                }
                for pname in property_names:
                    if property_granularity.get(pname) == "residue":
                        # Correlated with latent dim 0 + noise
                        row[pname] = float(
                            rec.latents[res_idx, 0] * 0.8
                            + rng.standard_normal() * 0.5
                        )
                    else:
                        # Protein-level: same for all residues
                        row[pname] = float(
                            mean_lat[1] * 0.7 + rng.standard_normal() * 0.3
                        )
                rows.append(row)
        else:
            # All protein-level
            row = {
                "protein_id": rec.protein_id,
                "residue_index": None,
            }
            for pname in property_names:
                row[pname] = float(
                    mean_lat[1] * 0.7 + rng.standard_normal() * 0.3
                )
            rows.append(row)

    return pd.DataFrame(rows)


def make_steering_decisions(
    probe_results: pd.DataFrame,
    goodhart_flagged: set[str],
    decisions_cfg: dict,
) -> pd.DataFrame:
    """Apply go/no-go decision logic per property.

    Parameters
    ----------
    probe_results : pd.DataFrame
        Full probe results table.
    goodhart_flagged : set[str]
        Properties flagged as Goodhart controls.
    decisions_cfg : dict
        Threshold config: steerable_r2_min, steerable_knn_gap_max,
        nonlinear_knn_r2_min, nonlinear_linear_r2_max, drop_r2_max.

    Returns
    -------
    pd.DataFrame
        Columns: property, decision, best_linear_r2, best_mlp_r2,
        best_knn_r2, note.
    """
    rows = []
    for prop in probe_results["property"].unique():
        prop_df = probe_results[probe_results["property"] == prop]

        best_linear = prop_df[prop_df["probe_type"] == "linear"]["r2_mean"].max()
        best_mlp = prop_df[prop_df["probe_type"] == "mlp"]["r2_mean"].max()
        best_knn = prop_df[prop_df["probe_type"] == "knn"]["r2_mean"].max()
        best_learnable = max(best_linear, best_mlp)

        note = ""

        # Special case: Rg
        if prop == "rg":
            note = ("Rg is backbone-determined; latent-only probes should be "
                    "near-zero unless Rg correlates with an encoded property.")

        # Goodhart check
        if prop in goodhart_flagged:
            decision = "goodhart_control"
            note += " Flagged as Goodhart control — evaluation only."
        elif best_learnable >= decisions_cfg["steerable_r2_min"]:
            knn_gap = best_knn - best_learnable
            if knn_gap < decisions_cfg["steerable_knn_gap_max"]:
                decision = "steerable"
            else:
                decision = "nonlinear_encoded"
                note += f" kNN gap = {knn_gap:.2f} exceeds threshold."
        elif (best_knn >= decisions_cfg["nonlinear_knn_r2_min"]
              and best_linear < decisions_cfg["nonlinear_linear_r2_max"]):
            decision = "nonlinear_encoded"
        elif max(best_linear, best_mlp, best_knn) < decisions_cfg["drop_r2_max"]:
            decision = "drop"
        else:
            decision = "ambiguous"
            note += " Does not cleanly fit any decision category."

        rows.append({
            "property": prop,
            "decision": decision,
            "best_linear_r2": float(best_linear),
            "best_mlp_r2": float(best_mlp),
            "best_knn_r2": float(best_knn),
            "note": note.strip(),
        })

    return pd.DataFrame(rows)


def run(config: dict, use_synthetic: bool = False) -> None:
    """Execute all Part 2 analyses.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.
    use_synthetic : bool
        If True, generate synthetic data for smoke testing.
    """
    apply_style()

    seed = config["seed"]
    rng = np.random.default_rng(seed)

    fig_dir = config["outputs"]["figures_dir"]
    tab_dir = config["outputs"]["tables_dir"]
    cache_dir = config["outputs"]["cache_dir"]
    out_dir = config["outputs"]["base_dir"]
    dpi = config["outputs"]["figure_dpi"]
    formats = config["outputs"]["figure_formats"]
    p2 = config["part2"]

    property_names = p2["property_names"]
    property_granularity = p2["property_granularity"]

    # --- Load data ---
    if use_synthetic:
        logger.info("Using synthetic dataset")
        records = make_synthetic_dataset(
            n_proteins=30, length=50, rng=rng, include_coords=True,
        )
        prop_df = _make_synthetic_properties(
            records, property_names, property_granularity, rng,
        )
    else:
        data_cfg = config["data"]
        length_range = data_cfg.get("length_range")
        if length_range is not None:
            length_range = tuple(length_range)
        records = load_dataset(
            latent_dir=data_cfg["latent_dir"],
            file_format=data_cfg["file_format"],
            field_names=data_cfg["field_names"],
            load_coords=True,
            subsample=data_cfg.get("subsample"),
            rng=rng,
            length_range=length_range,
        )
        prop_df = load_properties(
            p2["property_file"], property_names, property_granularity,
        )

    # --- Step 1: Align ---
    logger.info("Step 1: Aligning properties to latents")
    records, prop_df = align_properties_to_latents(
        records, prop_df, property_names, property_granularity,
    )
    n_proteins = len(records)
    logger.info("Aligned: %d proteins", n_proteins)

    # Cache aligned dataset
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    try:
        prop_df.to_parquet(Path(cache_dir) / "aligned_dataset.parquet", index=False)
    except ImportError:
        logger.info("pyarrow not available, caching as CSV")
        prop_df.to_csv(Path(cache_dir) / "aligned_dataset.csv", index=False)

    # --- Step 2: Property-property correlations ---
    logger.info("Step 2: Property-property correlations")
    available_props = [p for p in property_names if p in prop_df.columns]

    pearson_pp, spearman_pp = compute_property_correlations(
        prop_df, available_props, property_granularity,
    )
    save_table(pearson_pp.reset_index(), "property_pearson", tab_dir)
    save_table(spearman_pp.reset_index(), "property_spearman", tab_dir)

    if len(available_props) >= 2:
        plot_clustered_heatmap(pearson_pp, figures_dir=fig_dir, dpi=dpi, formats=formats)
        plt.close("all")

    clusters = cluster_properties(pearson_pp, threshold=p2["corr_cluster_threshold"])
    save_table(clusters, "property_clusters", tab_dir)

    goodhart_df = find_goodhart_pairs(
        pearson_pp, p2["goodhart_pairs"], p2["goodhart_divergence_threshold"],
    )
    save_table(goodhart_df, "goodhart_pairs", tab_dir)
    goodhart_flagged = set()
    for _, row in goodhart_df.iterrows():
        if row.get("diverges", False):
            goodhart_flagged.add(row["prop_a"])
            goodhart_flagged.add(row["prop_b"])

    # --- Steps 3-6: Probes at multiple noise levels and input variants ---
    logger.info("Steps 3-6: Running probes")
    logger.info("  t-convention: t=0 pure noise, t=1 clean (code convention)")

    noise_levels = p2["noise_levels"]
    n_noise_samples = p2["noise_samples_per_protein"]
    input_variants = p2["input_variants"]
    probe_cfg = p2["probes"]

    all_probe_results = []

    for prop_name in available_props:
        gran = property_granularity.get(prop_name, "protein")
        logger.info("Property: %s (granularity: %s)", prop_name, gran)

        for t in noise_levels:
            logger.info("  t=%.1f (code convention: 0=noise, 1=clean)", t)

            # Generate noised latents
            if t == 1.0:
                noised_dict = None  # use clean latents
            else:
                noised_dict = {}
                for rec in records:
                    child_rng = np.random.default_rng(
                        rng.integers(0, 2**31)
                    )
                    noised_dict[rec.protein_id] = make_noised_latents(
                        rec.latents, t, n_noise_samples, child_rng,
                    )

            for variant in input_variants:
                # Skip latent_plus_backbone if no coords
                if variant == "latent_plus_backbone" and any(
                    r.ca_coords is None for r in records
                ):
                    logger.warning("Skipping %s — no CA coords", variant)
                    continue

                try:
                    X, y, groups = prepare_probe_inputs(
                        records, prop_df, prop_name, gran, variant,
                        noised_latents=noised_dict,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to prepare inputs for %s/%s/t=%.1f: %s",
                        prop_name, variant, t, e,
                    )
                    continue

                if len(X) < 10:
                    logger.warning(
                        "Too few samples for %s/%s/t=%.1f: %d",
                        prop_name, variant, t, len(X),
                    )
                    continue

                probe_results = run_all_probes(
                    X, y, groups, probe_cfg,
                    rng_seed=seed,
                )

                for res in probe_results:
                    res["property"] = prop_name
                    res["t"] = t
                    res["t_convention"] = "code (0=noise, 1=clean)"
                    res["input_variant"] = variant
                    all_probe_results.append(res)

    # --- Step 7: Results table ---
    logger.info("Step 7: Building results table")
    results_df = pd.DataFrame(all_probe_results)

    # Drop fold_r2s list column for the saved table
    results_save = results_df.drop(columns=["fold_r2s"], errors="ignore")
    try:
        save_table(results_save, "probe_results", tab_dir, fmt="parquet")
    except (ImportError, Exception):
        save_table(results_save, "probe_results", tab_dir, fmt="csv")

    # Wide pivot for human reading
    if len(results_save) > 0:
        pivot = results_save.pivot_table(
            index=["property", "t", "input_variant"],
            columns="probe_type",
            values="r2_mean",
        ).reset_index()
        save_table(pivot, "probe_results_wide", tab_dir, fmt="csv")

    # --- Step 8: Decisions ---
    logger.info("Step 8: Steering decisions")
    decisions = make_steering_decisions(
        results_df, goodhart_flagged, p2["decisions"],
    )
    save_table(decisions, "steering_decisions", tab_dir)
    logger.info("Decisions:\n%s", decisions.to_string())

    # --- Step 9: UMAP ---
    logger.info("Step 9: UMAP visualization")
    try:
        import umap as _umap_check  # noqa: F401

        umap_cfg = p2["umap"]
        pooled, umap_ids = pool_for_umap(records, mode=umap_cfg["pooling"])
        embedding = run_umap(
            pooled,
            n_neighbors=umap_cfg["n_neighbors"],
            min_dist=umap_cfg["min_dist"],
            metric=umap_cfg["metric"],
            seed=seed,
        )

        # Best R² per property for annotation
        best_r2 = {}
        if len(results_df) > 0:
            for prop in available_props:
                prop_mask = results_df["property"] == prop
                if prop_mask.any():
                    best_r2[prop] = float(results_df.loc[prop_mask, "r2_mean"].max())

        plot_umap_grid(
            embedding, umap_ids, prop_df, available_props, best_r2,
            percentile_clip=umap_cfg["percentile_clip"],
            figures_dir=fig_dir, dpi=dpi, formats=formats,
        )
        plt.close("all")
    except ImportError:
        logger.warning("umap-learn not installed — skipping UMAP visualization. "
                       "Install with: pip install umap-learn")

    # --- Metadata ---
    write_run_metadata(out_dir, config, dataset_size=n_proteins, extra={
        "part": "part2_property_probes",
        "n_properties": len(available_props),
        "noise_levels": noise_levels,
        "t_convention": "code (0=noise, 1=clean)",
    })

    # --- Summary ---
    summary = _build_summary(decisions, available_props, goodhart_df, results_df)
    summary_path = Path(out_dir) / "part2_summary.md"
    summary_path.write_text(summary)
    logger.info("Part 2 complete. Summary: %s", summary_path)


def _build_summary(
    decisions: pd.DataFrame,
    available_props: list[str],
    goodhart_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> str:
    """Build a markdown summary of Part 2 results."""
    # Count decisions
    counts = decisions["decision"].value_counts().to_dict()

    lines = [
        "# Part 2: Property Probes Summary",
        "",
        "## t-convention",
        "All probe results use code convention: t=0 is pure noise, t=1 is clean data.",
        "WARNING: Steering predictor convention has NOT been verified.",
        "",
        "## Decision summary",
        "",
        f"- Properties analyzed: {len(available_props)}",
    ]
    for cat in ["steerable", "nonlinear_encoded", "goodhart_control", "drop", "ambiguous"]:
        lines.append(f"- {cat}: {counts.get(cat, 0)}")

    lines.extend(["", "## Decisions", ""])
    lines.append(decisions.to_string(index=False))

    if len(goodhart_df) > 0:
        lines.extend(["", "## Goodhart pairs", ""])
        lines.append(goodhart_df.to_string(index=False))

    if len(results_df) > 0:
        lines.extend(["", "## Top probe results (by R²)", ""])
        top = (
            results_df
            .sort_values("r2_mean", ascending=False)
            .head(15)
            [["property", "t", "input_variant", "probe_type", "r2_mean", "r2_std"]]
        )
        lines.append(top.to_string(index=False))

    lines.extend([
        "",
        "## Figures",
        "- `figures/property_correlation_clustered.{png,pdf}` — clustered property heatmap",
        "- `figures/umap_property_grid.{png,pdf}` — 3x3 UMAP grid",
        "",
        "## Tables",
        "- `tables/probe_results.parquet` — full probe results",
        "- `tables/probe_results_wide.csv` — pivot for human reading",
        "- `tables/steering_decisions.csv` — go/no-go per property",
        "- `tables/property_clusters.csv` — correlation clusters",
        "- `tables/goodhart_pairs.csv` — Goodhart pair analysis",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 2: Property probes")
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
