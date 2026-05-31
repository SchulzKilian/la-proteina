"""Dataset for the concept-bottleneck predictor.

Same latents/splits/stats pipeline as the no-bottleneck predictor (reuses ZScoreStats,
LengthBucketBatchSampler, create_held_out_split, create_fold_assignments from dataset.py),
but each record additionally carries the bottleneck supervision targets:
  - residue_type [L]        (amino-acid identity, OpenFold restype order)
  - torsion_sin_cos [L,7,2] (omega, phi, psi, chi1-4) from OpenFold atom37_to_torsion_angles
  - torsion_mask [L,7]      (1 where that torsion is defined for the residue)

Torsions are computed once at load time from coords_nm/coord_mask/residue_type, and the bulky
full-atom coordinates are then discarded (only the compact torsion targets are kept in memory).
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# openfold is vendored in the main la-proteina repo root (parents[3]), not pip-installed.
_MAIN_REPO = str(Path(__file__).resolve().parents[3])
if _MAIN_REPO not in sys.path:
    sys.path.insert(0, _MAIN_REPO)
from openfold.data.data_transforms import atom37_to_torsion_angles

from .dataset import PROPERTY_NAMES  # 14 property names, shared with the no-bottleneck model

logger = logging.getLogger(__name__)

N_TORSIONS = 7
AA_PAD_IDX = -100  # cross_entropy ignore_index for padded residues


@dataclass
class CBMProteinRecord:
    protein_id: str
    latents: np.ndarray          # [L, 8] float32
    residue_type: np.ndarray     # [L] int64 (0..20)
    torsion_sin_cos: np.ndarray  # [L, 7, 2] float32
    torsion_mask: np.ndarray     # [L, 7] float32
    length: int


def _compute_torsions(coords_nm: torch.Tensor, coord_mask: torch.Tensor,
                      residue_type: torch.Tensor):
    """coords_nm [L,37,3], coord_mask [L,37], residue_type [L] -> (sin_cos [L,7,2], mask [L,7])."""
    protein = {
        "aatype": residue_type.long().clamp(max=20).unsqueeze(0),       # [1, L]
        "all_atom_positions": coords_nm.float().unsqueeze(0),           # [1, L, 37, 3]
        "all_atom_mask": coord_mask.float().unsqueeze(0),               # [1, L, 37]
    }
    out = atom37_to_torsion_angles()(protein)  # @curry1: pass prefix-args (none), then protein
    sin_cos = out["torsion_angles_sin_cos"][0].float()  # [L, 7, 2]
    tmask = out["torsion_angles_mask"][0].float()       # [L, 7]
    return sin_cos, tmask


def load_cbm_dataset(latent_dir, length_range=None):
    """Mirror of src.data.loader.load_dataset but loads AA + torsion bottleneck targets.

    Returns records sorted by protein_id (so splits match the no-bottleneck pipeline).
    """
    latent_dir = Path(latent_dir)
    if not latent_dir.is_dir():
        raise FileNotFoundError(f"Latent directory not found: {latent_dir}")
    files = sorted(latent_dir.rglob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files in {latent_dir}")
    logger.info("Found %d .pt files; loading + computing torsions ...", len(files))

    from tqdm import tqdm
    records, n_failed = [], 0
    lo, hi = (length_range if length_range else (0, 10 ** 9))
    for path in tqdm(files, desc="Loading + torsions", disable=len(files) < 100):
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)

            def g(k):
                return d[k] if isinstance(d, dict) else getattr(d, k)

            latents = torch.as_tensor(g("mean")).float()          # [L, 8]
            L = latents.shape[0]
            if not (lo <= L <= hi):
                continue
            residue_type = torch.as_tensor(g("residue_type")).long().clamp(max=20)  # [L]
            sin_cos, tmask = _compute_torsions(
                torch.as_tensor(g("coords_nm")), torch.as_tensor(g("coord_mask")), residue_type
            )
            records.append(CBMProteinRecord(
                protein_id=str(g("id")),
                latents=latents.numpy().astype(np.float32),
                residue_type=residue_type.numpy().astype(np.int64),
                torsion_sin_cos=sin_cos.numpy().astype(np.float32),
                torsion_mask=tmask.numpy().astype(np.float32),
                length=int(L),
            ))
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            logger.warning("Failed to load %s: %s", path, e)

    if n_failed:
        logger.warning("Failed to load %d / %d files", n_failed, len(files))
    records.sort(key=lambda r: r.protein_id)
    logger.info("Loaded %d CBM records (%d residues)", len(records), sum(r.length for r in records))
    return records


class CBMPropertyDataset(Dataset):
    """Returns latents + z-scored property targets + AA/torsion bottleneck targets."""

    def __init__(self, records, prop_df, stats=None, t_value: float = 1.0):
        self.records = records
        self.stats = stats
        self.t_value = t_value
        prop_by_id = {}
        for _, row in prop_df.iterrows():
            prop_by_id[row["protein_id"]] = np.array(
                [row.get(p, np.nan) for p in PROPERTY_NAMES], dtype=np.float32
            )
        self.prop_by_id = prop_by_id

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        targets = self.prop_by_id[rec.protein_id].copy()
        if self.stats is not None:
            targets = self.stats.transform(targets)
        return {
            "latents": torch.from_numpy(rec.latents),
            "residue_type": torch.from_numpy(rec.residue_type),
            "torsion_sin_cos": torch.from_numpy(rec.torsion_sin_cos),
            "torsion_mask": torch.from_numpy(rec.torsion_mask),
            "targets": torch.from_numpy(targets),
            "length": rec.length,
            "protein_id": rec.protein_id,
            "t": self.t_value,
        }


def cbm_collate_fn(batch):
    max_len = max(b["length"] for b in batch)
    B = len(batch)
    D = batch[0]["latents"].shape[1]

    latents = torch.zeros(B, max_len, D, dtype=torch.float32)
    residue_type = torch.full((B, max_len), AA_PAD_IDX, dtype=torch.long)
    torsion_sin_cos = torch.zeros(B, max_len, N_TORSIONS, 2, dtype=torch.float32)
    torsion_mask = torch.zeros(B, max_len, N_TORSIONS, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    targets = torch.stack([b["targets"] for b in batch])
    t = torch.tensor([b["t"] for b in batch], dtype=torch.float32)

    for i, b in enumerate(batch):
        L = b["length"]
        latents[i, :L] = b["latents"]
        residue_type[i, :L] = b["residue_type"]
        torsion_sin_cos[i, :L] = b["torsion_sin_cos"]
        torsion_mask[i, :L] = b["torsion_mask"]
        mask[i, :L] = True

    return {
        "latents": latents,
        "residue_type": residue_type,
        "torsion_sin_cos": torsion_sin_cos,
        "torsion_mask": torsion_mask,
        "mask": mask,
        "targets": targets,
        "t": t,
        "protein_ids": [b["protein_id"] for b in batch],
    }
