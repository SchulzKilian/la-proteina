"""Wrapper that loads the trained PropertyTransformer + normalisation stats from checkpoint."""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

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
    mean: np.ndarray  # [13]
    std: np.ndarray   # [13]


class SteeringPredictor(nn.Module):
    """Loads a trained PropertyTransformer checkpoint and exposes predict/gradient methods.

    The checkpoint is expected to contain:
        model_state_dict: the model weights
        stats_mean: np.ndarray [13]
        stats_std: np.ndarray [13]
    """

    def __init__(self, checkpoint_path: str, device: torch.device | str = "cpu"):
        super().__init__()
        self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load z-score stats
        self.stats = ZScoreStats(
            mean=np.array(ckpt["stats_mean"], dtype=np.float32),
            std=np.array(ckpt["stats_std"], dtype=np.float32),
        )
        # Register as buffers for device transfer
        self.register_buffer("_stats_mean", torch.from_numpy(self.stats.mean))
        self.register_buffer("_stats_std", torch.from_numpy(self.stats.std))

        # n_properties is inferred from the checkpoint's stats_mean length —
        # newer checkpoints (e.g. logs/multitask_t1/20260427_161809) ship 14
        # heads (camsol_intrinsic added at idx 13); older ones ship 13.
        n_properties = int(self.stats.mean.shape[0])

        # Reconstruct model architecture from checkpoint or defaults
        self.model = PropertyTransformer(
            latent_dim=8,
            d_model=128,
            n_heads=4,
            n_layers=3,
            ffn_expansion=4,
            dropout=0.1,
            n_properties=n_properties,
            max_len=1024,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Freeze all parameters — we never train this
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Move whole module (incl. stats buffers) to device together.
        # (model.to() alone leaves _stats_mean/_std on CPU → device mismatch.)
        self.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        z_clean: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning de-normalised predictions.

        Args:
            z_clean: [B, L, 8] latent vectors (clean estimate)
            mask: [B, L] bool
            t: [B] float, defaults to 1.0 (clean data)

        Returns:
            [B, 13] de-normalised property predictions
        """
        B = z_clean.shape[0]
        if t is None:
            t = torch.ones(B, device=z_clean.device)

        preds_zscore = self.model(z_clean, mask, t)  # [B, 13]
        # De-normalise
        return preds_zscore * self._stats_std + self._stats_mean

    def predict_with_grad(
        self,
        z_clean: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass WITH gradients enabled (for backprop through z_clean).

        Returns z-scored predictions (not de-normalised) to keep gradient scale
        consistent across properties.

        Args:
            z_clean: [B, L, 8] — must have requires_grad=True
            mask: [B, L] bool
            t: [B] float

        Returns:
            [B, 13] z-scored predictions
        """
        B = z_clean.shape[0]
        if t is None:
            t = torch.ones(B, device=z_clean.device)

        return self.model(z_clean, mask, t)  # [B, 13] z-scored
