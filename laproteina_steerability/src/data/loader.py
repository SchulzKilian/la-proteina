"""Dataset loader for cached protein latent representations."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ProteinRecord:
    """A single protein's cached encoder output.

    Attributes
    ----------
    protein_id : str
        Unique identifier for the protein.
    latents : np.ndarray
        Latent vectors, shape ``[L, latent_dim]`` float32.
    ca_coords : np.ndarray or None
        Alpha-carbon coordinates in nm, shape ``[L, 3]`` float32.
        None if not loaded.
    length : int
        Number of residues.
    """

    protein_id: str
    latents: np.ndarray
    ca_coords: np.ndarray | None
    length: int


def _load_pt(path: Path, field_names: dict[str, str], load_coords: bool) -> ProteinRecord:
    """Load a single .pt file into a ProteinRecord.

    Parameters
    ----------
    path : Path
        Path to the .pt file.
    field_names : dict
        Mapping of semantic names to keys in the file.
    load_coords : bool
        Whether to load CA coordinates.

    Returns
    -------
    ProteinRecord
    """
    data = torch.load(path, map_location="cpu", weights_only=False)

    # Handle both dict and object-with-attributes (e.g. PyG Data)
    def _get(key: str) -> Any:
        if isinstance(data, dict):
            return data[key]
        return getattr(data, key)

    latents = np.asarray(_get(field_names["latents"]), dtype=np.float32)
    length_key = field_names.get("length")
    length = int(_get(length_key)) if length_key else latents.shape[0]

    protein_id_raw = _get(field_names["protein_id"])
    protein_id = str(protein_id_raw)

    ca_coords = None
    if load_coords:
        raw_coords = np.asarray(_get(field_names["ca_coords"]), dtype=np.float32)
        # If coords are full-atom [L, N_atoms, 3], extract CA via ca_atom_index
        ca_idx = field_names.get("ca_atom_index")
        if ca_idx is not None and raw_coords.ndim == 3:
            ca_coords = raw_coords[:, int(ca_idx), :]  # [L, 3]
        elif raw_coords.ndim == 2:
            ca_coords = raw_coords  # already [L, 3]
        else:
            raise ValueError(
                f"ca_coords has shape {raw_coords.shape} but no ca_atom_index "
                f"specified to extract CA from full-atom coordinates"
            )

    return ProteinRecord(
        protein_id=protein_id,
        latents=latents,
        ca_coords=ca_coords,
        length=length,
    )


def _load_npz(path: Path, field_names: dict[str, str], load_coords: bool) -> ProteinRecord:
    """Load a single .npz file into a ProteinRecord.

    Parameters
    ----------
    path : Path
        Path to the .npz file.
    field_names : dict
        Mapping of semantic names to keys in the file.
    load_coords : bool
        Whether to load CA coordinates.

    Returns
    -------
    ProteinRecord
    """
    data = np.load(path, allow_pickle=True)

    latents = data[field_names["latents"]].astype(np.float32)
    length = int(data[field_names["length"]]) if field_names.get("length") else latents.shape[0]
    protein_id = str(data[field_names["protein_id"]])

    ca_coords = None
    if load_coords:
        ca_coords = data[field_names["ca_coords"]].astype(np.float32)

    return ProteinRecord(
        protein_id=protein_id,
        latents=latents,
        ca_coords=ca_coords,
        length=length,
    )


_LOADERS = {
    "pt": _load_pt,
    "npz": _load_npz,
}


def load_dataset(
    latent_dir: str | Path,
    file_format: str,
    field_names: dict[str, str],
    load_coords: bool = False,
    subsample: int | None = None,
    rng: np.random.Generator | None = None,
    length_range: tuple[int, int] | None = None,
) -> list[ProteinRecord]:
    """Load all protein records from a directory of cached encoder outputs.

    Parameters
    ----------
    latent_dir : str or Path
        Directory containing per-protein files.
    file_format : str
        File extension: "pt" or "npz".
    field_names : dict
        Mapping from semantic field names (latents, ca_coords, protein_id,
        length) to the actual keys used in the files. Also supports
        ``ca_atom_index`` (int) to extract CA from full-atom coords.
    load_coords : bool
        Whether to load CA coordinates (needed for Part 2's
        latent_plus_backbone variant).
    subsample : int or None
        If set, randomly subsample this many proteins (after length filter).
    rng : np.random.Generator or None
        Random generator for subsampling. Required if subsample is set.
    length_range : tuple[int, int] or None
        If set, only keep proteins with length in ``[min, max]`` inclusive.

    Returns
    -------
    list[ProteinRecord]
        Loaded protein records, sorted by protein_id for reproducibility.
    """
    latent_dir = Path(latent_dir)
    if not latent_dir.is_dir():
        raise FileNotFoundError(f"Latent directory not found: {latent_dir}")

    loader_fn = _LOADERS.get(file_format)
    if loader_fn is None:
        raise ValueError(f"Unsupported file format: {file_format}. Supported: {list(_LOADERS.keys())}")

    # Collect all files (may be in subdirectories for sharded layouts)
    logger.info("Scanning for .%s files in %s ...", file_format, latent_dir)
    files = sorted(latent_dir.rglob(f"*.{file_format}"))
    if not files:
        raise FileNotFoundError(f"No .{file_format} files found in {latent_dir}")

    logger.info("Found %d .%s files", len(files), file_format)

    # Pre-subsample files when dataset is large and subsample is requested,
    # to avoid loading hundreds of thousands of files we'll discard.
    # Over-sample by 2x to leave room for length filtering and load failures.
    if subsample is not None and subsample < len(files):
        if rng is None:
            raise ValueError("rng is required when subsample is set")
        pre_n = min(len(files), subsample * 2)
        indices = rng.choice(len(files), size=pre_n, replace=False)
        files = [files[i] for i in sorted(indices)]
        logger.info("Pre-subsampled to %d files for loading", len(files))

    from tqdm import tqdm
    records: list[ProteinRecord] = []
    n_failed = 0
    for path in tqdm(files, desc="Loading proteins", disable=len(files) < 100):
        try:
            rec = loader_fn(path, field_names, load_coords)
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

    # Subsample (after length filter so the filter doesn't shrink a pre-subsampled set)
    if subsample is not None and subsample < len(records):
        if rng is None:
            raise ValueError("rng is required when subsample is set")
        indices = rng.choice(len(records), size=subsample, replace=False)
        records = [records[i] for i in sorted(indices)]
        logger.info("Subsampled to %d proteins", len(records))

    records.sort(key=lambda r: r.protein_id)
    logger.info("Loaded %d proteins (%d total residues)",
                len(records), sum(r.length for r in records))
    return records


def pool_latents(records: list[ProteinRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Pool all per-residue latents into a single array.

    Parameters
    ----------
    records : list[ProteinRecord]
        Loaded protein records.

    Returns
    -------
    all_latents : np.ndarray
        Shape ``[N_total_residues, latent_dim]``.
    protein_ids : np.ndarray
        Shape ``[N_total_residues]``, dtype object. The protein_id for each
        residue row.
    """
    latent_chunks = []
    id_chunks = []
    for rec in records:
        latent_chunks.append(rec.latents)
        id_chunks.append(np.full(rec.length, rec.protein_id, dtype=object))
    return np.concatenate(latent_chunks, axis=0), np.concatenate(id_chunks, axis=0)


def make_synthetic_dataset(
    n_proteins: int = 10,
    length: int = 50,
    latent_dim: int = 8,
    rng: np.random.Generator | None = None,
    include_coords: bool = True,
) -> list[ProteinRecord]:
    """Create a synthetic dataset for smoke testing.

    Latents are drawn from a multivariate normal with a random covariance
    (not identity — tests should catch code that assumes independence).
    CA coordinates are random points in a 3D box.

    Parameters
    ----------
    n_proteins : int
        Number of proteins to generate.
    length : int
        Number of residues per protein (fixed for simplicity).
    latent_dim : int
        Dimensionality of latent vectors.
    rng : np.random.Generator or None
        Random generator. Defaults to Generator(PCG64(0)).
    include_coords : bool
        Whether to generate CA coordinates.

    Returns
    -------
    list[ProteinRecord]
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Random covariance for latents
    A = rng.standard_normal((latent_dim, latent_dim))
    cov = A @ A.T / latent_dim
    mean = rng.standard_normal(latent_dim) * 0.5

    records = []
    for i in range(n_proteins):
        L = length + rng.integers(-10, 10)  # slight length variation
        L = max(L, 10)
        latents = rng.multivariate_normal(mean, cov, size=L).astype(np.float32)
        ca_coords = rng.standard_normal((L, 3)).astype(np.float32) * 0.5 if include_coords else None
        records.append(ProteinRecord(
            protein_id=f"synth_{i:04d}",
            latents=latents,
            ca_coords=ca_coords,
            length=L,
        ))
    return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    records = make_synthetic_dataset(n_proteins=10, length=50)
    print(f"Created {len(records)} synthetic proteins")
    all_lat, all_ids = pool_latents(records)
    print(f"Pooled latents shape: {all_lat.shape}")
    print(f"Unique protein IDs: {len(np.unique(all_ids))}")
    print(f"Latent mean: {all_lat.mean(0)}")
    print(f"Latent std:  {all_lat.std(0)}")
