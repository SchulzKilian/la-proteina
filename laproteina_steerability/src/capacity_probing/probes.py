"""Simple, non-Transformer probe architectures for capacity sweep.

Each probe takes per-residue latents [B, L, D] + mask [B, L] and outputs [B, 13].
All probes pool to a single per-protein vector before the head (so they differ
in non-linear-ness, not in pooling scheme — isolates the "how entangled is the
info in the latent?" axis).

Included sizes span 4 orders of magnitude in parameter count, so you can trace
a true capacity vs R² curve. The 3-layer Transformer from multitask_predictor
fills the upper end of the axis separately.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B, L, D] × [B, L] → [B, D] mean over valid positions."""
    mask_f = mask.float().unsqueeze(-1)
    return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


class LinearProbe(nn.Module):
    """Linear: pool → linear head. Smallest probe. ~100 params."""

    def __init__(self, latent_dim: int = 8, n_properties: int = 13):
        super().__init__()
        self.head = nn.Linear(latent_dim, n_properties)

    def forward(self, x, mask, t=None):
        return self.head(_mean_pool(x, mask))


class MLPProbe(nn.Module):
    """Pool → MLP. Varies capacity via depth × width."""

    def __init__(
        self,
        latent_dim: int = 8,
        hidden: int = 64,
        n_layers: int = 1,
        n_properties: int = 13,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden
        layers += [nn.Linear(in_dim, n_properties)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, mask, t=None):
        return self.net(_mean_pool(x, mask))


class PerResidueMLPProbe(nn.Module):
    """Per-residue MLP → mean-pool → linear head.

    Strictly more expressive than MLPProbe at matched param count because each
    residue gets non-linearly transformed before pooling (captures per-residue
    interactions with neighbours at the pool step rather than only at input).
    """

    def __init__(
        self,
        latent_dim: int = 8,
        hidden: int = 128,
        n_layers: int = 2,
        n_properties: int = 13,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden
        self.residue_net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_properties)

    def forward(self, x, mask, t=None):
        h = self.residue_net(x)  # [B, L, hidden]
        return self.head(_mean_pool(h, mask))


def build_probe(config: dict) -> nn.Module:
    """Build a probe from a config dict.

    Config keys:
        kind: "linear" | "mlp" | "per_residue_mlp"
        hidden, n_layers, dropout: optional, kind-dependent
    """
    kind = config["kind"]
    kwargs = {k: v for k, v in config.items() if k not in ("kind", "name")}
    if kind == "linear":
        return LinearProbe(**kwargs)
    if kind == "mlp":
        return MLPProbe(**kwargs)
    if kind == "per_residue_mlp":
        return PerResidueMLPProbe(**kwargs)
    raise ValueError(f"Unknown probe kind: {kind}")


PROBE_CONFIGS = [
    # (name, config). Ordered by approximate parameter count.
    {"name": "linear",              "kind": "linear"},
    {"name": "mlp_h32_L1",          "kind": "mlp", "hidden": 32, "n_layers": 1},
    {"name": "mlp_h64_L1",          "kind": "mlp", "hidden": 64, "n_layers": 1},
    {"name": "mlp_h128_L2",         "kind": "mlp", "hidden": 128, "n_layers": 2},
    {"name": "per_res_mlp_h64_L1",  "kind": "per_residue_mlp", "hidden": 64, "n_layers": 1},
    {"name": "per_res_mlp_h128_L2", "kind": "per_residue_mlp", "hidden": 128, "n_layers": 2},
    {"name": "per_res_mlp_h256_L3", "kind": "per_residue_mlp", "hidden": 256, "n_layers": 3},
]
