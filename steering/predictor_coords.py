"""Wrapper that loads CA-conditioned PropertyTransformerCoords ckpts.

Mirrors `steering.predictor.SteeringPredictor` but takes `(z, ca, mask, t)`
on every call. The ckpt format is the one written by
`multitask_predictor_coords.train_coords.train_fold`:

  {
    "model_state_dict": ...,
    "arch_kwargs":      {...},    # kwargs to rebuild PropertyTransformerCoords
    "stats_mean":       [P],
    "stats_std":        [P],
    ...
  }

`arch_kwargs` is preferred when present. As a fall-back (e.g. for an old
hand-saved ckpt) the loader reconstructs with the default kwargs.

The existing latent-only `SteeringPredictor` is untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

# Add steerability src to path so we can import the coord-aware model
_STEERABILITY_ROOT = Path(__file__).resolve().parents[1] / "laproteina_steerability"
if str(_STEERABILITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STEERABILITY_ROOT))

from src.multitask_predictor_coords.model_coords import PropertyTransformerCoords


@dataclass
class ZScoreStats:
    mean: np.ndarray
    std: np.ndarray


class SteeringPredictorCoords(nn.Module):
    """Loads one or more PropertyTransformerCoords ckpts. predict/predict_with_grad
    take (z, ca, mask, t)."""

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

        stats_means: List[np.ndarray] = []
        stats_stds: List[np.ndarray] = []
        models: List[nn.Module] = []
        n_props_first: int | None = None
        for cp in ckpt_paths:
            ckpt = torch.load(cp, map_location=self.device, weights_only=False)

            if not ckpt.get("coord_conditioned", False):
                raise ValueError(
                    f"Checkpoint {cp} is not marked coord_conditioned=True. "
                    "Use steering.predictor.SteeringPredictor for latent-only ckpts."
                )

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

            arch = ckpt.get("arch_kwargs")
            if arch is None:
                arch = dict(
                    latent_dim=8, d_model=128, n_heads=4, n_layers=3,
                    ffn_expansion=4, dropout=0.1,
                    n_properties=n_props_first, max_len=1024,
                )
            else:
                arch = dict(arch)
                arch.setdefault("n_properties", n_props_first)

            m = PropertyTransformerCoords(**arch)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            models.append(m)

        self.stats = ZScoreStats(
            mean=np.stack(stats_means).mean(0),
            std=np.stack(stats_stds).mean(0),
        )
        self.register_buffer("_stats_mean", torch.from_numpy(self.stats.mean))
        self.register_buffer("_stats_std", torch.from_numpy(self.stats.std))

        self.models = nn.ModuleList(models)
        self.to(self.device)

    def _ensemble_zscore(
        self,
        z: torch.Tensor,
        ca: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if self.n_folds == 1:
            return self.models[0](z, ca, mask, t)
        preds = [m(z, ca, mask, t) for m in self.models]
        return torch.stack(preds, dim=0).mean(dim=0)

    @torch.no_grad()
    def predict(
        self,
        z_clean: torch.Tensor,
        ca: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = z_clean.shape[0]
        if t is None:
            t = torch.ones(B, device=z_clean.device)
        preds_zscore = self._ensemble_zscore(z_clean, ca, mask, t)
        return preds_zscore * self._stats_std + self._stats_mean

    def predict_with_grad(
        self,
        z_clean: torch.Tensor,
        ca: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = z_clean.shape[0]
        if t is None:
            t = torch.ones(B, device=z_clean.device)
        return self._ensemble_zscore(z_clean, ca, mask, t)
