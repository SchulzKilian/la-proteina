"""Wrapper that loads the trained PropertyTransformer + normalisation stats from checkpoint.

Supports two modes:
  - single fold: pass a single checkpoint path (string).
  - ensemble: pass a list of checkpoint paths. Predictions are the mean of the
    z-scored outputs across folds; stats are the per-property mean of each
    fold's z-score stats (folds trained on overlapping but different splits
    so stats differ slightly — averaging avoids preferring fold_0 by accident).
    Used to harden steering against gradient hacking — adversarial directions
    that fool a single fold tend to cancel under ensemble averaging.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

# Add steerability src to path so we can import the model
_STEERABILITY_ROOT = Path(__file__).resolve().parents[1] / "laproteina_steerability"
if str(_STEERABILITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STEERABILITY_ROOT))

from src.multitask_predictor.model import PropertyTransformer


@dataclass
class ZScoreStats:
    """Per-property mean and std for de/normalisation."""
    mean: np.ndarray  # [P]
    std: np.ndarray   # [P]


class SteeringPredictor(nn.Module):
    """Loads one or more PropertyTransformer checkpoints and exposes predict/gradient methods."""

    def __init__(
        self,
        checkpoint_path: Union[str, Sequence[str]],
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)

        if isinstance(checkpoint_path, (str, Path)):
            ckpt_paths: List[str] = [str(checkpoint_path)]
        else:
            ckpt_paths = [str(p) for p in checkpoint_path]
        self.n_folds = len(ckpt_paths)
        if self.n_folds == 0:
            raise ValueError("checkpoint_path is empty")

        # Load each fold; collect stats; build PropertyTransformer instances
        stats_means: List[np.ndarray] = []
        stats_stds: List[np.ndarray] = []
        models: List[nn.Module] = []
        n_props_first: int | None = None
        for cp in ckpt_paths:
            ckpt = torch.load(cp, map_location=self.device, weights_only=False)
            sm = np.array(ckpt["stats_mean"], dtype=np.float32)
            ss = np.array(ckpt["stats_std"], dtype=np.float32)
            if n_props_first is None:
                n_props_first = int(sm.shape[0])
            elif sm.shape[0] != n_props_first:
                raise ValueError(
                    f"Inconsistent n_properties across folds: {n_props_first} vs {sm.shape[0]} ({cp})"
                )
            stats_means.append(sm)
            stats_stds.append(ss)

            m = PropertyTransformer(
                latent_dim=8, d_model=128, n_heads=4, n_layers=3,
                ffn_expansion=4, dropout=0.1, n_properties=n_props_first, max_len=1024,
            )
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            models.append(m)

        # Stats: per-property mean across folds. (Folds trained on slightly
        # different CV splits so their normalisers are not byte-identical;
        # averaging keeps the de-normalisation symmetric across folds.)
        self.stats = ZScoreStats(
            mean=np.stack(stats_means).mean(0),
            std=np.stack(stats_stds).mean(0),
        )
        self.register_buffer("_stats_mean", torch.from_numpy(self.stats.mean))
        self.register_buffer("_stats_std", torch.from_numpy(self.stats.std))

        # nn.ModuleList so .to(device) and .eval() reach all folds
        self.models = nn.ModuleList(models)

        # Move whole module (incl. stats buffers) to device together.
        # (model.to() alone leaves _stats_mean/_std on CPU → device mismatch.)
        self.to(self.device)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _ensemble_zscore(self, z: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Mean z-scored prediction across all loaded folds. Shape [B, P]."""
        if self.n_folds == 1:
            return self.models[0](z, mask, t)
        preds = [m(z, mask, t) for m in self.models]
        return torch.stack(preds, dim=0).mean(dim=0)

    @torch.no_grad()
    def predict(
        self,
        z_clean: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning de-normalised ensemble-mean predictions.

        Args:
            z_clean: [B, L, 8] latent vectors (clean estimate)
            mask: [B, L] bool
            t: [B] float, defaults to 1.0 (clean data)

        Returns:
            [B, P] de-normalised property predictions (P = 13 or 14)
        """
        B = z_clean.shape[0]
        if t is None:
            t = torch.ones(B, device=z_clean.device)
        preds_zscore = self._ensemble_zscore(z_clean, mask, t)
        return preds_zscore * self._stats_std + self._stats_mean

    def predict_with_grad(
        self,
        z_clean: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass WITH gradients enabled (for backprop through z_clean).

        Returns z-scored ensemble-mean predictions (not de-normalised) to
        keep gradient scale consistent across properties.

        Args:
            z_clean: [B, L, 8] — must have requires_grad=True
            mask: [B, L] bool
            t: [B] float

        Returns:
            [B, P] z-scored ensemble-mean predictions.
        """
        B = z_clean.shape[0]
        if t is None:
            t = torch.ones(B, device=z_clean.device)
        return self._ensemble_zscore(z_clean, mask, t)
