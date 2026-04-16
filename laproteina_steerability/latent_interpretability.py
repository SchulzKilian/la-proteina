"""Per-residue latent interpretability: linear probes for structural and sequence features.

Trains linear probes to predict residue-level features from the 8D per-residue
latent vectors of La-Proteina's VAE encoder.

Features:
    - Residue identity (20-class multinomial logistic regression)
    - Distance to chain centre of mass (Ridge regression)
    - Neighbour count within 8 A (Ridge regression)

Run with:
    python -m latent_interpretability --config config/default.yaml
    python -m latent_interpretability --config config/default.yaml --synthetic
    python -m latent_interpretability --config config/default.yaml --subsample 500
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.common.io import save_figure, save_table, write_run_metadata
from src.common.plotting import CMAP_SEQUENTIAL, apply_style
from src.data.loader import load_dataset, make_synthetic_dataset, pool_latents

logger = logging.getLogger(__name__)

# 20 standard amino acids in OpenFold residue-type order
AA_LABELS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
]
N_AA = len(AA_LABELS)
NEIGHBOUR_RADIUS_NM = 0.8  # 8 A = 0.8 nm (coords are in nm)


# ---------------------------------------------------------------------------
# Extended data record (adds residue_type to the standard ProteinRecord)
# ---------------------------------------------------------------------------

@dataclass
class InterpRecord:
    """Protein record with residue types for interpretability analysis.

    Attributes
    ----------
    protein_id : str
        Unique identifier.
    latents : np.ndarray
        Shape ``[L, latent_dim]`` float32.
    ca_coords : np.ndarray
        Shape ``[L, 3]`` float32, in nm.
    residue_type : np.ndarray
        Shape ``[L]`` int64, amino acid index per residue.
    length : int
        Number of residues.
    """

    protein_id: str
    latents: np.ndarray
    ca_coords: np.ndarray
    residue_type: np.ndarray
    length: int


def _load_interp_record(
    path: Path,
    field_names: dict[str, str],
) -> InterpRecord:
    """Load a single .pt file into an InterpRecord.

    Extends the standard loader to also read residue_type.
    """
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)

    def _get(key: str) -> Any:
        if isinstance(data, dict):
            return data[key]
        return getattr(data, key)

    latents = np.asarray(_get(field_names["latents"]), dtype=np.float32)
    length = latents.shape[0]

    protein_id = str(_get(field_names["protein_id"]))

    # CA coordinates
    raw_coords = np.asarray(_get(field_names["ca_coords"]), dtype=np.float32)
    ca_idx = field_names.get("ca_atom_index")
    if ca_idx is not None and raw_coords.ndim == 3:
        ca_coords = raw_coords[:, int(ca_idx), :]
    elif raw_coords.ndim == 2:
        ca_coords = raw_coords
    else:
        raise ValueError(
            f"ca_coords has shape {raw_coords.shape} but no ca_atom_index "
            f"specified to extract CA from full-atom coordinates"
        )

    # Residue type
    residue_type = np.asarray(_get("residue_type"), dtype=np.int64)

    return InterpRecord(
        protein_id=protein_id,
        latents=latents,
        ca_coords=ca_coords,
        residue_type=residue_type,
        length=length,
    )


def load_interp_dataset(
    config: dict,
    subsample: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[InterpRecord]:
    """Load dataset with residue types, using the same config as steerability.

    Parameters
    ----------
    config : dict
        Full config dict (uses ``data`` section).
    subsample : int or None
        Override subsample count.
    rng : np.random.Generator
        Random generator for subsampling.

    Returns
    -------
    list[InterpRecord]
    """
    data_cfg = config["data"]
    latent_dir = Path(data_cfg["latent_dir"])
    field_names = data_cfg["field_names"]
    length_range = data_cfg.get("length_range")
    if length_range is not None:
        length_range = tuple(length_range)
    sub = subsample if subsample is not None else data_cfg.get("subsample")

    logger.info("Scanning for .pt files in %s ...", latent_dir)
    files = sorted(latent_dir.rglob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {latent_dir}")
    logger.info("Found %d .pt files", len(files))

    # Pre-subsample to avoid loading the full 355K files
    if sub is not None and sub < len(files):
        if rng is None:
            raise ValueError("rng is required when subsample is set")
        pre_n = min(len(files), sub * 2)
        indices = rng.choice(len(files), size=pre_n, replace=False)
        files = [files[i] for i in sorted(indices)]
        logger.info("Pre-subsampled to %d files for loading", len(files))

    from tqdm import tqdm

    records: list[InterpRecord] = []
    n_failed = 0
    for path in tqdm(files, desc="Loading proteins", disable=len(files) < 100):
        try:
            rec = _load_interp_record(path, field_names)
            records.append(rec)
        except Exception as e:
            n_failed += 1
            logger.warning("Failed to load %s: %s", path, e)

    if n_failed > 0:
        logger.warning("Failed to load %d / %d files", n_failed, len(files))

    # Length filter
    if length_range is not None:
        lo, hi = length_range
        before = len(records)
        records = [r for r in records if lo <= r.length <= hi]
        logger.info("Length filter [%d, %d]: kept %d / %d proteins",
                     lo, hi, len(records), before)

    # Final subsample
    if sub is not None and sub < len(records):
        indices = rng.choice(len(records), size=sub, replace=False)
        records = [records[i] for i in sorted(indices)]
        logger.info("Subsampled to %d proteins", len(records))

    records.sort(key=lambda r: r.protein_id)
    logger.info("Loaded %d proteins (%d total residues)",
                len(records), sum(r.length for r in records))
    return records


def make_synthetic_interp_dataset(
    n_proteins: int = 20,
    length: int = 80,
    latent_dim: int = 8,
    rng: np.random.Generator | None = None,
) -> list[InterpRecord]:
    """Create a synthetic dataset for smoke testing."""
    if rng is None:
        rng = np.random.default_rng(0)

    records = []
    for i in range(n_proteins):
        L = max(20, length + rng.integers(-20, 20))
        latents = rng.standard_normal((L, latent_dim)).astype(np.float32)
        ca_coords = rng.standard_normal((L, 3)).astype(np.float32) * 2.0
        residue_type = rng.integers(0, N_AA, size=L)
        records.append(InterpRecord(
            protein_id=f"synth_{i:04d}",
            latents=latents,
            ca_coords=ca_coords,
            residue_type=residue_type,
            length=L,
        ))
    return records


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_features(records: list[InterpRecord]) -> dict[str, np.ndarray]:
    """Compute per-residue features from the loaded dataset.

    Returns
    -------
    dict with keys:
        latents : np.ndarray, shape [N, latent_dim]
        residue_type : np.ndarray, shape [N], int64
        dist_to_com : np.ndarray, shape [N], float32
        neighbour_count : np.ndarray, shape [N], int32
        protein_ids : np.ndarray, shape [N], object (for grouped CV)
    """
    latent_chunks = []
    restype_chunks = []
    dist_chunks = []
    neighbour_chunks = []
    id_chunks = []

    for rec in records:
        latent_chunks.append(rec.latents)
        restype_chunks.append(rec.residue_type)
        id_chunks.append(np.full(rec.length, rec.protein_id, dtype=object))

        # Distance to chain centre of mass
        com = rec.ca_coords.mean(axis=0)  # [3]
        dists = np.linalg.norm(rec.ca_coords - com, axis=1)  # [L]
        dist_chunks.append(dists.astype(np.float32))

        # Neighbour count within radius
        tree = cKDTree(rec.ca_coords)
        counts = tree.query_ball_point(rec.ca_coords, r=NEIGHBOUR_RADIUS_NM,
                                       return_length=True)
        # query_ball_point includes self, subtract 1
        neighbour_chunks.append(np.asarray(counts, dtype=np.int32) - 1)

    return {
        "latents": np.concatenate(latent_chunks, axis=0),
        "residue_type": np.concatenate(restype_chunks, axis=0),
        "dist_to_com": np.concatenate(dist_chunks, axis=0),
        "neighbour_count": np.concatenate(neighbour_chunks, axis=0),
        "protein_ids": np.concatenate(id_chunks, axis=0),
    }


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _grouped_cv_regression(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Ridge regression with grouped cross-validation.

    Returns
    -------
    dict with r2_mean, r2_std, r2_per_fold, coefs (from last fold for weight heatmap)
    """
    gkf = GroupKFold(n_splits=n_folds)
    r2s = []
    last_coefs = None

    for train_idx, test_idx in gkf.split(X, y, groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        model = Ridge(alpha=alpha)
        model.fit(X_train, y[train_idx])
        y_pred = model.predict(X_test)
        r2s.append(r2_score(y[test_idx], y_pred))
        last_coefs = model.coef_

    return {
        "r2_mean": np.mean(r2s),
        "r2_std": np.std(r2s),
        "r2_per_fold": r2s,
        "coefs": last_coefs,  # [latent_dim]
    }


def _grouped_cv_classification(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Multinomial logistic regression with grouped cross-validation.

    Returns
    -------
    dict with accuracy_mean, accuracy_std, per_class_accuracy, coefs [n_classes, latent_dim]
    """
    gkf = GroupKFold(n_splits=n_folds)
    accuracies = []
    per_class_accs_list = []
    last_coefs = None

    for train_idx, test_idx in gkf.split(X, y, groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        model = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            C=1.0,
        )
        model.fit(X_train, y[train_idx])
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y[test_idx], y_pred))

        # Per-class accuracy
        per_class = {}
        for cls in range(N_AA):
            mask = y[test_idx] == cls
            if mask.sum() > 0:
                per_class[cls] = accuracy_score(y[test_idx][mask], y_pred[mask])
            else:
                per_class[cls] = np.nan
        per_class_accs_list.append(per_class)
        last_coefs = model.coef_  # [n_classes, latent_dim]

    # Average per-class accuracy across folds
    avg_per_class = {}
    for cls in range(N_AA):
        vals = [fold[cls] for fold in per_class_accs_list if not np.isnan(fold[cls])]
        avg_per_class[cls] = np.mean(vals) if vals else np.nan

    return {
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "per_class_accuracy": avg_per_class,
        "coefs": last_coefs,  # [n_classes, latent_dim]
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_weight_heatmap(
    ridge_coefs: dict[str, np.ndarray],
    logreg_coefs: np.ndarray,
    latent_dim: int,
    figures_dir: str | Path,
    dpi: int = 300,
    formats: list[str] | None = None,
) -> None:
    """Weight heatmap: 8 dims x features, colored by absolute weight magnitude.

    Rows: latent dims (D0-D7). Columns: features.
    Left panel: scalar features (dist_to_com, neighbour_count).
    Right panel: residue identity (20 amino acids).
    """
    # --- Combined weight matrix ---
    # Scalar features: coefs are [latent_dim], transpose to [latent_dim, n_features]
    scalar_names = list(ridge_coefs.keys())
    scalar_W = np.column_stack([ridge_coefs[name] for name in scalar_names])

    # Classification: coefs are [n_classes, latent_dim], transpose to [latent_dim, n_classes]
    class_W = logreg_coefs.T  # [latent_dim, 20]

    # Full matrix: [latent_dim, n_scalar + 20]
    W = np.concatenate([scalar_W, class_W], axis=1)
    col_labels = scalar_names + AA_LABELS
    row_labels = [f"D{i}" for i in range(latent_dim)]

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(np.abs(W), aspect="auto", cmap=CMAP_SEQUENTIAL, interpolation="nearest")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(latent_dim))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Latent dimension")
    ax.set_title("Probe weight magnitude (|w|)")

    # Separator line between scalar and classification features
    ax.axvline(x=len(scalar_names) - 0.5, color="white", linewidth=2, linestyle="--")

    plt.colorbar(im, ax=ax, label="|weight|", shrink=0.8)
    save_figure(fig, "weight_heatmap", figures_dir, dpi=dpi, formats=formats)
    plt.close(fig)


def plot_top_dim_per_aa(
    logreg_coefs: np.ndarray,
    figures_dir: str | Path,
    dpi: int = 300,
    formats: list[str] | None = None,
) -> None:
    """Bar chart showing which latent dim has the highest weight for each AA."""
    # logreg_coefs: [20, latent_dim]
    top_dims = np.argmax(np.abs(logreg_coefs), axis=1)  # [20]
    latent_dim = logreg_coefs.shape[1]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10(top_dims / max(latent_dim - 1, 1))
    bars = ax.bar(range(N_AA), np.abs(logreg_coefs[np.arange(N_AA), top_dims]),
                  color=colors)
    ax.set_xticks(range(N_AA))
    ax.set_xticklabels(AA_LABELS, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("|weight| of top dim")
    ax.set_title("Highest-weight latent dimension per amino acid")

    # Annotate with dim number
    for i, (bar, dim) in enumerate(zip(bars, top_dims)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"D{dim}", ha="center", va="bottom", fontsize=8)

    save_figure(fig, "top_dim_per_aa", figures_dir, dpi=dpi, formats=formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run(config: dict, use_synthetic: bool = False, subsample: int | None = None) -> None:
    """Execute latent interpretability analysis.

    Parameters
    ----------
    config : dict
        Full configuration dictionary (same format as steerability).
    use_synthetic : bool
        If True, use synthetic data for smoke testing.
    subsample : int or None
        Override subsample count from CLI.
    """
    apply_style()

    seed = config["seed"]
    rng = np.random.default_rng(seed)

    fig_dir = config["outputs"]["figures_dir"]
    tab_dir = config["outputs"]["tables_dir"]
    out_dir = config["outputs"]["base_dir"]
    dpi = config["outputs"]["figure_dpi"]
    formats = config["outputs"]["figure_formats"]
    n_folds = config.get("part2", {}).get("probes", {}).get("cv_folds", 5)

    # --- Load data ---
    if use_synthetic:
        logger.info("Using synthetic dataset")
        records = make_synthetic_interp_dataset(n_proteins=20, length=80, rng=rng)
    else:
        records = load_interp_dataset(config, subsample=subsample, rng=rng)

    if not records:
        raise RuntimeError("No proteins loaded")

    # --- Compute features ---
    logger.info("Computing per-residue features ...")
    feats = compute_features(records)
    latents = feats["latents"]
    protein_ids = feats["protein_ids"]
    n_residues, latent_dim = latents.shape
    n_proteins = len(records)
    logger.info("Feature matrix: %d residues x %d latent dims from %d proteins",
                n_residues, latent_dim, n_proteins)

    # --- Filter to standard amino acids (0-19) ---
    valid_mask = feats["residue_type"] < N_AA
    n_excluded = (~valid_mask).sum()
    if n_excluded > 0:
        logger.info("Excluding %d residues with non-standard residue type", n_excluded)
        latents = latents[valid_mask]
        protein_ids = protein_ids[valid_mask]
        for key in ("residue_type", "dist_to_com", "neighbour_count"):
            feats[key] = feats[key][valid_mask]
        n_residues = latents.shape[0]

    # =====================================================================
    # Probe 1: Residue identity (multinomial logistic regression)
    # =====================================================================
    logger.info("Training residue identity probe (logistic regression, %d-fold CV) ...",
                n_folds)
    cls_result = _grouped_cv_classification(
        latents, feats["residue_type"], protein_ids, n_folds=n_folds,
    )

    print("\n" + "=" * 60)
    print("RESIDUE IDENTITY (multinomial logistic regression)")
    print("=" * 60)
    print(f"  Overall accuracy: {cls_result['accuracy_mean']:.3f} "
          f"+/- {cls_result['accuracy_std']:.3f}")
    print(f"  (chance = {1/N_AA:.3f})")
    print()
    print("  Per-class accuracy:")
    for cls_idx in range(N_AA):
        acc = cls_result["per_class_accuracy"][cls_idx]
        print(f"    {AA_LABELS[cls_idx]:>3s}: {acc:.3f}" if not np.isnan(acc)
              else f"    {AA_LABELS[cls_idx]:>3s}: N/A (no samples)")

    # Top dim per AA
    if cls_result["coefs"] is not None:
        print()
        print("  Highest-weight latent dim per amino acid:")
        top_dims = np.argmax(np.abs(cls_result["coefs"]), axis=1)
        for cls_idx in range(N_AA):
            print(f"    {AA_LABELS[cls_idx]:>3s} -> D{top_dims[cls_idx]}")

    # =====================================================================
    # Probe 2: Distance to centre of mass (Ridge)
    # =====================================================================
    logger.info("Training distance-to-COM probe (Ridge, %d-fold CV) ...", n_folds)
    dist_result = _grouped_cv_regression(
        latents, feats["dist_to_com"], protein_ids, n_folds=n_folds,
    )

    print("\n" + "=" * 60)
    print("DISTANCE TO CENTRE OF MASS (Ridge regression)")
    print("=" * 60)
    print(f"  R^2: {dist_result['r2_mean']:.3f} +/- {dist_result['r2_std']:.3f}")

    # =====================================================================
    # Probe 3: Neighbour count (Ridge)
    # =====================================================================
    logger.info("Training neighbour count probe (Ridge, %d-fold CV) ...", n_folds)
    neigh_result = _grouped_cv_regression(
        latents, feats["neighbour_count"].astype(np.float32), protein_ids,
        n_folds=n_folds,
    )

    print("\n" + "=" * 60)
    print("NEIGHBOUR COUNT within 8 A (Ridge regression)")
    print("=" * 60)
    print(f"  R^2: {neigh_result['r2_mean']:.3f} +/- {neigh_result['r2_std']:.3f}")

    # =====================================================================
    # Outputs
    # =====================================================================

    # --- Results table ---
    rows = [
        {"feature": "residue_identity", "metric": "accuracy",
         "value": cls_result["accuracy_mean"], "std": cls_result["accuracy_std"],
         "probe": "logistic_regression"},
        {"feature": "dist_to_com", "metric": "R2",
         "value": dist_result["r2_mean"], "std": dist_result["r2_std"],
         "probe": "ridge"},
        {"feature": "neighbour_count", "metric": "R2",
         "value": neigh_result["r2_mean"], "std": neigh_result["r2_std"],
         "probe": "ridge"},
    ]
    results_df = pd.DataFrame(rows)
    save_table(results_df, "interpretability_probe_results", tab_dir)

    # Per-class accuracy table
    per_class_rows = []
    for cls_idx in range(N_AA):
        per_class_rows.append({
            "aa": AA_LABELS[cls_idx],
            "accuracy": cls_result["per_class_accuracy"][cls_idx],
            "top_dim": int(top_dims[cls_idx]) if cls_result["coefs"] is not None else None,
        })
    per_class_df = pd.DataFrame(per_class_rows)
    save_table(per_class_df, "interpretability_per_class_accuracy", tab_dir)

    # --- Figures ---
    if cls_result["coefs"] is not None:
        ridge_coefs = {
            "dist_to_com": dist_result["coefs"],
            "neighbour_count": neigh_result["coefs"],
        }
        plot_weight_heatmap(
            ridge_coefs, cls_result["coefs"], latent_dim,
            figures_dir=fig_dir, dpi=dpi, formats=formats,
        )
        plot_top_dim_per_aa(
            cls_result["coefs"], figures_dir=fig_dir, dpi=dpi, formats=formats,
        )

    # --- Metadata ---
    write_run_metadata(out_dir, config, dataset_size=n_proteins, extra={
        "part": "latent_interpretability",
        "n_residues": n_residues,
        "latent_dim": latent_dim,
        "features": ["residue_identity", "dist_to_com", "neighbour_count"],
    })

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Figures: {fig_dir}/weight_heatmap.*, {fig_dir}/top_dim_per_aa.*")
    print(f"  Tables:  {tab_dir}/interpretability_probe_results.csv")
    print(f"           {tab_dir}/interpretability_per_class_accuracy.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latent interpretability: linear probes for per-residue features",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for smoke testing")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Override subsample count (number of proteins)")
    args = parser.parse_args()

    config = _load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config.get("log_level", "INFO")),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    run(config, use_synthetic=args.synthetic, subsample=args.subsample)


if __name__ == "__main__":
    main()
