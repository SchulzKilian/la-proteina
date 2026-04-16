"""Dataset and length-bucketed batch sampler for multi-task property prediction."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from src.data.loader import ProteinRecord

logger = logging.getLogger(__name__)

PROPERTY_NAMES = [
    "swi", "tango", "net_charge", "pI", "iupred3",
    "iupred3_fraction_disordered", "shannon_entropy",
    "hydrophobic_patch_total_area", "hydrophobic_patch_n_large",
    "sap", "scm_positive", "scm_negative", "rg",
]


@dataclass
class ZScoreStats:
    """Per-property mean and std computed on training data."""
    mean: np.ndarray   # [13]
    std: np.ndarray    # [13]

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean

    @classmethod
    def fit(cls, y: np.ndarray) -> ZScoreStats:
        """Fit on [N, 13] target array, handling NaN per column."""
        mean = np.nanmean(y, axis=0)
        std = np.nanstd(y, axis=0)
        std[std < 1e-8] = 1.0  # avoid div-by-zero for constant properties
        return cls(mean=mean, std=std)


class PropertyDataset(Dataset):
    """Wraps ProteinRecord list + aligned property DataFrame for PyTorch.

    Each item returns:
        latents: [L, 8] float32
        targets: [13] float32 (z-scored if stats provided, NaN for missing)
        length: int
        protein_id: str
    """

    def __init__(
        self,
        records: list[ProteinRecord],
        prop_df: pd.DataFrame,
        stats: ZScoreStats | None = None,
        t_value: float = 1.0,
    ):
        self.records = records
        self.t_value = t_value
        self.stats = stats

        # Build protein_id -> property vector lookup
        prop_by_id = {}
        for _, row in prop_df.iterrows():
            pid = row["protein_id"]
            vals = np.array([row.get(p, np.nan) for p in PROPERTY_NAMES], dtype=np.float32)
            prop_by_id[pid] = vals
        self.prop_by_id = prop_by_id

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        latents = torch.from_numpy(rec.latents)  # [L, 8]
        targets = self.prop_by_id[rec.protein_id].copy()  # [13]

        if self.stats is not None:
            targets = self.stats.transform(targets)

        return {
            "latents": latents,
            "targets": torch.from_numpy(targets),
            "length": rec.length,
            "protein_id": rec.protein_id,
            "t": self.t_value,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad to batch max length, produce attention mask."""
    max_len = max(b["length"] for b in batch)
    B = len(batch)
    D = batch[0]["latents"].shape[1]

    latents = torch.zeros(B, max_len, D, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    targets = torch.stack([b["targets"] for b in batch])  # [B, 13]
    t = torch.tensor([b["t"] for b in batch], dtype=torch.float32)  # [B]

    for i, b in enumerate(batch):
        L = b["length"]
        latents[i, :L] = b["latents"]
        mask[i, :L] = True

    return {
        "latents": latents,
        "mask": mask,
        "targets": targets,
        "t": t,
        "protein_ids": [b["protein_id"] for b in batch],
    }


class LengthBucketBatchSampler(Sampler):
    """Sort by length, form contiguous buckets, shuffle within and across buckets.

    For validation, set shuffle=False for deterministic ordering.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        # Sort indices by length
        sorted_indices = np.argsort(lengths)

        # Form batches from contiguous length ranges
        self.batches = []
        for start in range(0, len(sorted_indices), batch_size):
            self.batches.append(sorted_indices[start:start + batch_size].tolist())

    def __iter__(self):
        if self.shuffle:
            # Shuffle batch order
            order = self.rng.permutation(len(self.batches))
            for idx in order:
                batch = self.batches[idx]
                # Shuffle within batch
                yield list(self.rng.permutation(batch))
        else:
            for batch in self.batches:
                yield batch

    def __len__(self) -> int:
        return len(self.batches)


def create_held_out_split(
    records: list[ProteinRecord],
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split protein IDs into train (90%) and held-out test (10%).

    Stratified by length decile for balanced representation.

    Returns (train_ids, test_ids), both sorted.
    """
    rng = np.random.default_rng(seed)

    ids = np.array([r.protein_id for r in records])
    lengths = np.array([r.length for r in records])

    # Compute length deciles
    deciles = pd.qcut(lengths, q=10, labels=False, duplicates="drop")

    test_ids = []
    train_ids = []

    for dec in np.unique(deciles):
        dec_mask = deciles == dec
        dec_ids = ids[dec_mask]
        n_test = max(1, int(len(dec_ids) * test_fraction))
        chosen = rng.choice(len(dec_ids), size=n_test, replace=False)
        chosen_set = set(chosen)
        for i, pid in enumerate(dec_ids):
            if i in chosen_set:
                test_ids.append(pid)
            else:
                train_ids.append(pid)

    return sorted(train_ids), sorted(test_ids)


def create_fold_assignments(
    train_ids: list[str],
    n_folds: int = 5,
) -> pd.DataFrame:
    """Deterministic fold assignment using GroupKFold logic on sorted protein IDs.

    Returns DataFrame with columns: protein_id, fold.
    """
    from sklearn.model_selection import GroupKFold

    train_ids_sorted = sorted(train_ids)
    n = len(train_ids_sorted)

    # GroupKFold needs X, y, groups — we use dummy X/y, groups = arange
    X_dummy = np.zeros((n, 1))
    y_dummy = np.zeros(n)
    groups = np.arange(n)

    gkf = GroupKFold(n_splits=n_folds)
    fold_map = np.full(n, -1, dtype=np.int32)

    for fold_idx, (_, val_idx) in enumerate(gkf.split(X_dummy, y_dummy, groups)):
        fold_map[val_idx] = fold_idx

    return pd.DataFrame({
        "protein_id": train_ids_sorted,
        "fold": fold_map,
    })
