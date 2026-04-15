"""Property loading, alignment, and normalization for backbone inputs."""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from src.data.loader import ProteinRecord

logger = logging.getLogger(__name__)


def load_properties(
    property_file: str,
    property_names: list[str],
    property_granularity: dict[str, str],
) -> pd.DataFrame:
    """Load property table from parquet.

    Parameters
    ----------
    property_file : str
        Path to the parquet file with columns: protein_id,
        residue_index (nullable), and one column per property.
    property_names : list[str]
        Properties to load.
    property_granularity : dict[str, str]
        Maps property name to "protein" or "residue".

    Returns
    -------
    pd.DataFrame
        Loaded properties with columns: protein_id, residue_index,
        and one column per property.
    """
    df = pd.read_parquet(property_file)

    missing_cols = [p for p in property_names if p not in df.columns]
    if missing_cols:
        logger.warning("Properties not found in file, will be NaN: %s", missing_cols)

    required = ["protein_id"]
    if any(g == "residue" for g in property_granularity.values()):
        required.append("residue_index")

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not in property file")

    keep_cols = required + [p for p in property_names if p in df.columns]
    return df[keep_cols].copy()


def align_properties_to_latents(
    records: list[ProteinRecord],
    prop_df: pd.DataFrame,
    property_names: list[str],
    property_granularity: dict[str, str],
) -> tuple[list[ProteinRecord], pd.DataFrame]:
    """Align property table to latent records, dropping mismatches.

    For protein-level properties, ensures protein_ids match.
    For residue-level properties, additionally validates residue counts.

    Parameters
    ----------
    records : list[ProteinRecord]
        Loaded protein records.
    prop_df : pd.DataFrame
        Property table from ``load_properties``.
    property_names : list[str]
        Property columns to include.
    property_granularity : dict[str, str]
        Maps property name to "protein" or "residue".

    Returns
    -------
    records_aligned : list[ProteinRecord]
        Subset of records with matching properties.
    prop_aligned : pd.DataFrame
        Aligned property table.
    """
    latent_ids = {r.protein_id for r in records}
    prop_ids = set(prop_df["protein_id"].unique())

    common = latent_ids & prop_ids
    lat_only = latent_ids - prop_ids
    prop_only = prop_ids - latent_ids

    if lat_only:
        logger.warning("%d proteins in latents but not in properties — dropped", len(lat_only))
    if prop_only:
        logger.warning("%d proteins in properties but not in latents — dropped", len(prop_only))

    records_aligned = [r for r in records if r.protein_id in common]
    prop_aligned = prop_df[prop_df["protein_id"].isin(common)].copy()

    # Validate residue counts for residue-level properties
    has_residue_props = any(
        property_granularity.get(p) == "residue" for p in property_names
    )
    if has_residue_props and "residue_index" in prop_aligned.columns:
        rec_lengths = {r.protein_id: r.length for r in records_aligned}
        prop_counts = prop_aligned.groupby("protein_id").size().to_dict()
        mismatch = []
        for pid in list(common):
            if pid in prop_counts and pid in rec_lengths:
                if prop_counts[pid] != rec_lengths[pid]:
                    mismatch.append(pid)
        if mismatch:
            logger.warning(
                "%d proteins have residue count mismatch — dropped: %s",
                len(mismatch), mismatch[:5],
            )
            mismatch_set = set(mismatch)
            records_aligned = [r for r in records_aligned if r.protein_id not in mismatch_set]
            prop_aligned = prop_aligned[~prop_aligned["protein_id"].isin(mismatch_set)]

    logger.info("Aligned dataset: %d proteins", len(records_aligned))
    return records_aligned, prop_aligned


def normalize_backbone_se3(ca_coords: np.ndarray) -> np.ndarray:
    """SE(3)-normalize CA coordinates: center to COM, align to PCA axes.

    The rotation aligns the CA cloud to its principal axes (PCA).
    Sign convention: for each axis, the coordinate with the largest
    absolute value is made positive. This is a deterministic, cheap
    normalization that removes rigid-body degrees of freedom.

    Known limitation: near-degenerate CA clouds (e.g., nearly linear
    proteins) can produce discontinuous rotations. For a first pass
    this is acceptable; a Procrustes-based approach would be more
    robust but requires a reference structure.

    Parameters
    ----------
    ca_coords : np.ndarray
        Shape ``[L, 3]``, CA coordinates in nm.

    Returns
    -------
    np.ndarray
        Shape ``[L, 3]``, normalized coordinates.
    """
    # Center
    centered = ca_coords - ca_coords.mean(axis=0)

    # PCA rotation
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]

    # Rotate
    rotated = centered @ eigenvectors

    # Deterministic sign fix: largest |coord| on each axis is positive
    for ax in range(3):
        max_idx = np.argmax(np.abs(rotated[:, ax]))
        if rotated[max_idx, ax] < 0:
            rotated[:, ax] *= -1

    return rotated.astype(np.float32)


def prepare_probe_inputs(
    records: list[ProteinRecord],
    prop_df: pd.DataFrame,
    property_name: str,
    granularity: str,
    input_variant: Literal["latent_only", "latent_plus_backbone"],
    noised_latents: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare X, y arrays for probing.

    Parameters
    ----------
    records : list[ProteinRecord]
        Aligned protein records.
    prop_df : pd.DataFrame
        Aligned property table.
    property_name : str
        Which property column to use as target.
    granularity : str
        "protein" or "residue".
    input_variant : str
        "latent_only" or "latent_plus_backbone".
    noised_latents : dict[str, np.ndarray] or None
        If provided, maps protein_id -> noised latents ``[n_samples, L, D]``.
        If None, uses clean latents from records.

    Returns
    -------
    X : np.ndarray
        Input features. For protein-level: ``[N, D]`` or ``[N, D+3]``.
        For residue-level: ``[N_residues, D]`` or ``[N_residues, D+3]``.
    y : np.ndarray
        Target values. Shape ``[N]`` or ``[N_residues]``.
    groups : np.ndarray
        Group labels (protein index) for grouped CV. Same length as X.
    """
    X_parts = []
    y_parts = []
    group_parts = []

    prop_by_protein = {
        pid: grp for pid, grp in prop_df.groupby("protein_id")
    }

    for group_idx, rec in enumerate(records):
        pid = rec.protein_id
        if pid not in prop_by_protein:
            continue

        prop_rows = prop_by_protein[pid]
        if property_name not in prop_rows.columns:
            continue

        # Get latents (clean or noised)
        if noised_latents is not None and pid in noised_latents:
            # noised: [n_samples, L, D]
            lat_samples = noised_latents[pid]
        else:
            # clean: [1, L, D]
            lat_samples = rec.latents[np.newaxis]

        n_samples = lat_samples.shape[0]

        for s in range(n_samples):
            lat = lat_samples[s]  # [L, D]

            if input_variant == "latent_plus_backbone":
                if rec.ca_coords is None:
                    raise ValueError(f"CA coords required for latent_plus_backbone but missing for {pid}")
                coords_norm = normalize_backbone_se3(rec.ca_coords)
                features = np.concatenate([lat, coords_norm], axis=1)  # [L, D+3]
            else:
                features = lat  # [L, D]

            if granularity == "protein":
                # Mean-pool to protein level
                x_protein = features.mean(axis=0, keepdims=True)  # [1, D] or [1, D+3]
                prop_vals = prop_rows[property_name].values
                if len(prop_vals) == 1:
                    y_val = prop_vals[0]
                else:
                    # Multiple rows per protein for protein-level prop — take first
                    y_val = prop_vals[0]
                X_parts.append(x_protein)
                y_parts.append([y_val])
                group_parts.append([group_idx])
            else:
                # Residue level
                prop_vals = prop_rows[property_name].values
                if len(prop_vals) != lat.shape[0]:
                    logger.warning(
                        "Residue count mismatch for %s: latents=%d, props=%d — skipping",
                        pid, lat.shape[0], len(prop_vals),
                    )
                    continue
                X_parts.append(features)
                y_parts.append(prop_vals)
                group_parts.append(np.full(len(prop_vals), group_idx))

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.float32)
    groups = np.concatenate(group_parts, axis=0).astype(np.int64)

    # Drop NaN targets
    valid = ~np.isnan(y)
    if not valid.all():
        n_dropped = (~valid).sum()
        logger.warning("Dropped %d NaN target values for %s", n_dropped, property_name)
        X, y, groups = X[valid], y[valid], groups[valid]

    return X, y, groups


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Smoke test SE(3) normalization
    rng = np.random.default_rng(42)
    coords = rng.standard_normal((50, 3)).astype(np.float32)
    normed = normalize_backbone_se3(coords)
    print(f"Input COM: {coords.mean(0)}")
    print(f"Output COM: {normed.mean(0)}")
    print(f"Output cov diag: {np.cov(normed, rowvar=False).diagonal()}")
