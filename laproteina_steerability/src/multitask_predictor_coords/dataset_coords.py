"""Dataset + collate for the CA-conditioned multi-task property predictor.

Mirrors `multitask_predictor.dataset.PropertyDataset`, but also returns
`ca_coords[L, 3]` and pads them in the collate. Re-uses the existing
`PROPERTY_NAMES`, `ZScoreStats`, `LengthBucketBatchSampler`, and the
held-out / fold-assignment helpers — those are coord-agnostic.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.loader import ProteinRecord
from src.multitask_predictor.dataset import (
    PROPERTY_NAMES,
    ZScoreStats,
    LengthBucketBatchSampler,
    create_held_out_split,
    create_fold_assignments,
)


__all__ = [
    "PROPERTY_NAMES",
    "ZScoreStats",
    "LengthBucketBatchSampler",
    "create_held_out_split",
    "create_fold_assignments",
    "PropertyDatasetCoords",
    "collate_fn_coords",
]


class PropertyDatasetCoords(Dataset):
    """Same shape as PropertyDataset but each item also carries `coords[L, 3]`."""

    def __init__(
        self,
        records: list[ProteinRecord],
        prop_df,
        stats: ZScoreStats | None = None,
        t_value: float = 1.0,
    ):
        self.records = records
        self.t_value = t_value
        self.stats = stats

        prop_by_id = {}
        for _, row in prop_df.iterrows():
            pid = row["protein_id"]
            vals = np.array([row.get(p, np.nan) for p in PROPERTY_NAMES], dtype=np.float32)
            prop_by_id[pid] = vals
        self.prop_by_id = prop_by_id

        # Sanity: every record needs ca_coords populated.
        for r in records:
            if r.ca_coords is None:
                raise ValueError(
                    f"protein_id={r.protein_id} has no ca_coords. "
                    "Re-run loader with load_coords=True."
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        latents = torch.from_numpy(rec.latents)            # [L, 8]
        coords = torch.from_numpy(rec.ca_coords)           # [L, 3]
        targets = self.prop_by_id[rec.protein_id].copy()    # [P]

        if self.stats is not None:
            targets = self.stats.transform(targets)

        return {
            "latents": latents,
            "coords": coords,
            "targets": torch.from_numpy(targets),
            "length": rec.length,
            "protein_id": rec.protein_id,
            "t": self.t_value,
        }


def collate_fn_coords(batch: list[dict]) -> dict:
    """Pad to batch max length, produce attention mask, pad coords with zeros."""
    max_len = max(b["length"] for b in batch)
    B = len(batch)
    D_lat = batch[0]["latents"].shape[1]
    D_co = batch[0]["coords"].shape[1]

    latents = torch.zeros(B, max_len, D_lat, dtype=torch.float32)
    coords = torch.zeros(B, max_len, D_co, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    targets = torch.stack([b["targets"] for b in batch])
    t = torch.tensor([b["t"] for b in batch], dtype=torch.float32)

    for i, b in enumerate(batch):
        L = b["length"]
        latents[i, :L] = b["latents"]
        coords[i, :L] = b["coords"]
        mask[i, :L] = True

    return {
        "latents": latents,
        "coords": coords,
        "mask": mask,
        "targets": targets,
        "t": t,
        "protein_ids": [b["protein_id"] for b in batch],
    }
