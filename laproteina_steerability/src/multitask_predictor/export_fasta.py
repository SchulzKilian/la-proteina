"""Export protein sequences from .pt files as FASTA + metadata CSV.

Supports two modes:
  1. Scan a directory of .pt files (--pt-dir)
  2. Export only proteins listed in a properties CSV (--from-properties + --pt-dir)

Usage:
    # All proteins in a directory, filtered by length
    python -m src.multitask_predictor.export_fasta \
        --pt-dir /path/to/processed/ \
        --output-dir exports/ \
        --length-range 300 800

    # Only proteins in properties.csv (no length filter needed — CSV already filtered)
    python -m src.multitask_predictor.export_fasta \
        --pt-dir /path/to/processed/ \
        --from-properties data/properties.csv \
        --output-dir exports/camsol_batch
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# OpenFold residue type ordering (alphabetical by 3-letter code)
RESTYPES = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]
RESTYPE_WITH_X = RESTYPES + ["X"]

# 3-letter to 1-letter mapping for raw PDB .pt files that store `residues`
RESTYPE_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def residue_types_to_sequence(residue_type: np.ndarray) -> str:
    """Convert integer residue type indices to one-letter amino acid sequence."""
    return "".join(RESTYPE_WITH_X[int(i)] for i in residue_type)


def residues_3letter_to_sequence(residues: list[str]) -> str:
    """Convert 3-letter residue names to one-letter sequence."""
    return "".join(RESTYPE_3TO1.get(r, "X") for r in residues)


def _get_field(data, key):
    """Get a field from dict or PyG Data object."""
    if isinstance(data, dict):
        return data[key]
    return getattr(data, key)


def _has_field(data, key):
    """Check if field exists in dict or PyG Data object."""
    if isinstance(data, dict):
        return key in data
    return hasattr(data, key)


def _extract_sequence(data) -> str:
    """Extract amino acid sequence from a .pt data object.

    Tries in order:
      1. `residues` field (list of 3-letter codes, from raw PDB .pt files)
      2. `residue_type` field (integer indices, from latent .pt files)
    """
    if _has_field(data, "residues"):
        residues = _get_field(data, "residues")
        if isinstance(residues, (list, tuple)):
            return residues_3letter_to_sequence(residues)
    if _has_field(data, "residue_type"):
        rt = np.asarray(_get_field(data, "residue_type"), dtype=np.int64)
        return residue_types_to_sequence(rt)
    raise ValueError("No `residues` or `residue_type` field found")


def _extract_id(data) -> str:
    """Extract protein ID from a .pt data object."""
    for key in ("id", "name", "protein_id"):
        if _has_field(data, key):
            return str(_get_field(data, key))
    raise ValueError("No id/name/protein_id field found")


def _find_pt_file(protein_id: str, pt_dir: Path) -> Path | None:
    """Locate the .pt file for a protein_id in a sharded directory.

    Expects layout: <2-char-prefix>/<protein_id>.pt
    """
    prefix = protein_id[:2].lower()
    candidate = pt_dir / prefix / f"{protein_id}.pt"
    if candidate.exists():
        return candidate
    # Fallback: try flat layout
    flat = pt_dir / f"{protein_id}.pt"
    if flat.exists():
        return flat
    return None


def export(
    pt_dir: str | Path,
    output_dir: str | Path,
    length_range: tuple[int, int] | None = None,
    from_properties: str | Path | None = None,
) -> tuple[Path, Path]:
    """Export FASTA and metadata CSV from .pt files.

    Parameters
    ----------
    pt_dir : path
        Directory with .pt files (sharded or flat).
    output_dir : path
        Where to write output files.
    length_range : tuple or None
        (min_len, max_len) inclusive filter.
    from_properties : path or None
        If given, only export proteins listed in this CSV (must have protein_id column).

    Returns
    -------
    (fasta_path, csv_path)
    """
    pt_dir = Path(pt_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which files to process
    # First, scan the directory once to build a stem->path index (avoids per-file stat on Lustre)
    logger.info("Scanning %s for .pt files ...", pt_dir)
    all_files = sorted(pt_dir.rglob("*.pt"))
    stem_to_path = {f.stem: f for f in all_files}
    logger.info("Indexed %d .pt files", len(stem_to_path))

    if from_properties is not None:
        prop_df = pd.read_csv(from_properties)
        protein_ids = sorted(prop_df["protein_id"].unique())
        logger.info("Properties CSV has %d unique protein IDs", len(protein_ids))

        files = []
        missing = 0
        for pid in protein_ids:
            if pid in stem_to_path:
                files.append((pid, stem_to_path[pid]))
            else:
                missing += 1
        if missing > 0:
            logger.warning("%d protein IDs from CSV not found as .pt files", missing)
        logger.info("Matched %d proteins to .pt files", len(files))
    else:
        files = [(None, f) for f in all_files]

    fasta_path = output_dir / "proteins.fasta"
    csv_path = output_dir / "proteins_metadata.csv"

    n_written = 0
    n_skipped_length = 0
    n_failed = 0

    with open(fasta_path, "w") as fa, open(csv_path, "w") as csv_f:
        csv_f.write("protein_id,chain,sequence_length\n")

        for expected_id, path in tqdm(files, desc="Exporting"):
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)

                protein_id = expected_id if expected_id else _extract_id(data)
                sequence = _extract_sequence(data)
                seq_len = len(sequence)

                if length_range is not None:
                    lo, hi = length_range
                    if seq_len < lo or seq_len > hi:
                        n_skipped_length += 1
                        continue

                # Parse chain from protein_id (format: pdbid_chain, e.g. "101m_A")
                if "_" in protein_id:
                    parts = protein_id.rsplit("_", 1)
                    chain = parts[1] if len(parts) == 2 else ""
                else:
                    chain = ""

                fa.write(f">{protein_id}\n")
                for i in range(0, len(sequence), 80):
                    fa.write(sequence[i:i + 80] + "\n")

                csv_f.write(f"{protein_id},{chain},{seq_len}\n")
                n_written += 1

            except Exception as e:
                n_failed += 1
                if n_failed <= 5:
                    logger.warning("Failed to process %s: %s", path, e)

    logger.info(
        "Exported %d proteins, skipped %d (length filter), %d failed",
        n_written, n_skipped_length, n_failed,
    )
    logger.info("FASTA: %s (%.1f MB)", fasta_path, fasta_path.stat().st_size / 1e6)
    logger.info("CSV:   %s", csv_path)
    return fasta_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Export protein sequences as FASTA + CSV")
    parser.add_argument("--pt-dir", type=str, required=True,
                        help="Directory containing .pt files (sharded or flat)")
    parser.add_argument("--output-dir", type=str, default="exports",
                        help="Output directory for FASTA and CSV")
    parser.add_argument("--length-range", type=int, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Length range filter (inclusive)")
    parser.add_argument("--from-properties", type=str, default=None,
                        help="Only export proteins listed in this CSV (must have protein_id column)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    length_range = tuple(args.length_range) if args.length_range else None
    export(args.pt_dir, args.output_dir, length_range, args.from_properties)


if __name__ == "__main__":
    main()
