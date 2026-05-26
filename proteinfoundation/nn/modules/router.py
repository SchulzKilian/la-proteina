"""Move-1 top-K router: predicts dense's per-(layer, head, query) attention
pattern from cheap features.

Base architecture (~1.5M params at hidden_dim=128, score_dim=32):
  - Shared linear down-projection W_in: trunk_dim → hidden_dim, ReLU.
  - Optional 2-layer GELU MLP "trunk block" before the per-(L,H) projections
    (Probe 2 / bigger router).
  - Per-(layer, head) W_q, W_k ∈ [hidden_dim × score_dim].
  - Optional pair-distance feature: RBF expansion of CA-CA distance d_ij,
    additive per-(layer, head) scalar via W_pair ∈ R^{n_rbf}. Targets the
    spatial-attention layers. ~2.7k extra params.
  - Optional sinusoidal t-embedding (TopKRouterT subclass). Already shown
    in E066 to give a small constant gain (+0.05pp final) on its own.

Scores = (Q · K^T) / sqrt(score_dim)  [+ pair_bias_lh if pair_features].
NO softmax inside the module — the training loss does its own log-softmax
on the full-key softmax (mask padding to -inf first), and the eval code
does top-K.

Memory: forward output is [B, L, H, N, N]. For B=8 N=400 L=14 H=12 fp32
that's ~8.6 GB — dominant. At fp16 it's 4.3 GB. The RBF intermediate is
[B, N, N, n_rbf] ≈ 80 MB at B=8 N=400 n_rbf=16 fp32 — negligible.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    def __init__(
        self,
        trunk_dim: int = 768,
        hidden_dim: int = 128,
        score_dim: int = 32,
        n_layers: int = 14,
        n_heads: int = 12,
        pair_features: bool = False,
        n_rbf: int = 16,
        rbf_min_a: float = 0.0,
        rbf_max_a: float = 30.0,
        rbf_sigma_a: float = 2.0,
        mlp_block: bool = False,
        mlp_dim: int = 256,
    ):
        super().__init__()
        self.trunk_dim = trunk_dim
        self.hidden_dim = hidden_dim
        self.score_dim = score_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pair_features = pair_features
        self.mlp_block = mlp_block
        self.n_rbf = n_rbf
        self.rbf_min_a = float(rbf_min_a)
        self.rbf_max_a = float(rbf_max_a)
        self.rbf_sigma_a = float(rbf_sigma_a)

        self.W_in = nn.Linear(trunk_dim, hidden_dim)
        if mlp_block:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, hidden_dim),
                nn.GELU(),
            )
        else:
            self.mlp = None
        self.W_q = nn.Parameter(torch.empty(n_layers, n_heads, hidden_dim, score_dim))
        self.W_k = nn.Parameter(torch.empty(n_layers, n_heads, hidden_dim, score_dim))
        nn.init.normal_(self.W_q, std=0.02)
        nn.init.normal_(self.W_k, std=0.02)

        if pair_features:
            # Per-(layer, head) projection from RBF features to a scalar pair-bias.
            # Zero-init so init-time behavior is bias-free and the model has to
            # learn the spatial signal.
            self.W_pair = nn.Parameter(torch.zeros(n_layers, n_heads, n_rbf))
            # μ_k spaced linearly in Å.
            mu = torch.linspace(self.rbf_min_a, self.rbf_max_a, n_rbf)
            self.register_buffer("rbf_mu_a", mu)
        else:
            self.W_pair = None

    def _compute_rbf(self, coords_nm: torch.Tensor) -> torch.Tensor:
        """coords_nm: [B, N, 3] in nanometers. Returns [B, N, N, n_rbf] fp32.
        Convert nm → Å before RBF so the parameter scale matches the prompt's
        spec (μ ∈ 0..30 Å, σ = 2 Å).
        """
        # [B, N, 1, 3] − [B, 1, N, 3] → [B, N, N, 3]
        diff = coords_nm.unsqueeze(2) - coords_nm.unsqueeze(1)
        d_nm = torch.linalg.vector_norm(diff, dim=-1)              # [B, N, N]
        d_a = d_nm * 10.0                                          # nm → Å
        # [B, N, N, 1] vs [n_rbf]
        return torch.exp(
            -(d_a.unsqueeze(-1) - self.rbf_mu_a) ** 2
            / (2.0 * self.rbf_sigma_a ** 2)
        )

    def forward(
        self,
        layer_inputs: torch.Tensor,
        coords_nm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            layer_inputs: [B, n_layers, N, trunk_dim]
            coords_nm:    [B, N, 3] — required iff pair_features=True
        Returns:
            scores: [B, n_layers, n_heads, N, N]
        """
        assert layer_inputs.shape[1] == self.n_layers, (
            f"expected n_layers={self.n_layers}, got {layer_inputs.shape[1]}"
        )
        r = F.relu(self.W_in(layer_inputs))                        # [B, L, N, hid]
        if self.mlp is not None:
            r = self.mlp(r)                                        # [B, L, N, hid]
        q = torch.einsum("blnd,lhde->blhne", r, self.W_q)         # [B, L, H, N, score]
        k = torch.einsum("blnd,lhde->blhne", r, self.W_k)
        scores = torch.einsum("blhid,blhjd->blhij", q, k)         # [B, L, H, N, N]
        scores = scores / (self.score_dim ** 0.5)
        if self.pair_features:
            assert coords_nm is not None, \
                "pair_features=True requires coords_nm in forward()"
            rbf = self._compute_rbf(coords_nm)                    # [B, N, N, n_rbf]
            pair_bias = torch.einsum("bijk,lhk->blhij", rbf, self.W_pair)
            scores = scores + pair_bias
        return scores

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_one_layer(
        self,
        layer_input: torch.Tensor,
        layer_idx: int,
        coords_nm: torch.Tensor | None = None,
        precomputed_rbf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score one layer at a time (Move-2 trunk wiring).

        Args:
            layer_input: [B, N, trunk_dim] — input hidden state to trunk layer `layer_idx`.
            layer_idx:   int in [0, n_layers).
            coords_nm:   [B, N, 3] — required iff pair_features=True AND precomputed_rbf is None.
            precomputed_rbf: [B, N, N, n_rbf] — optional cached RBF (caller has already run
                _compute_rbf(coords_nm)). When given, coords_nm is ignored. Saves redundant
                _compute_rbf calls when this method is invoked per-layer.
        Returns:
            scores: [B, n_heads, N, N]
        """
        assert 0 <= layer_idx < self.n_layers, (
            f"layer_idx {layer_idx} out of range [0, {self.n_layers})"
        )
        r = F.relu(self.W_in(layer_input))                         # [B, N, hid]
        if self.mlp is not None:
            r = self.mlp(r)                                        # [B, N, hid]
        Wq_l = self.W_q[layer_idx]                                 # [H, hid, score]
        Wk_l = self.W_k[layer_idx]
        q = torch.einsum("bnd,hde->bhne", r, Wq_l)                 # [B, H, N, score]
        k = torch.einsum("bnd,hde->bhne", r, Wk_l)
        scores = torch.einsum("bhid,bhjd->bhij", q, k)             # [B, H, N, N]
        scores = scores / (self.score_dim ** 0.5)
        if self.pair_features:
            if precomputed_rbf is None:
                assert coords_nm is not None, \
                    "pair_features=True requires coords_nm or precomputed_rbf in forward_one_layer()"
                rbf = self._compute_rbf(coords_nm)                 # [B, N, N, n_rbf]
            else:
                rbf = precomputed_rbf
            Wp_l = self.W_pair[layer_idx]                          # [H, n_rbf]
            scores = scores + torch.einsum("bijk,hk->bhij", rbf, Wp_l)
        return scores


class TopKRouterT(TopKRouter):
    """Variant with explicit t-embedding conditioning.

    Targets the t=0.10 bucket where the un-conditioned router landed at 0.603
    (vs 0.669 at t=0.50). Sinusoidal embedding of the scalar t → small 2-layer
    MLP → broadcast-add to the post-W_in hidden state, before the per-(layer,
    head) Q/K projections. ~24k extra params.
    """
    def __init__(
        self,
        trunk_dim: int = 768,
        hidden_dim: int = 128,
        score_dim: int = 32,
        n_layers: int = 14,
        n_heads: int = 12,
        t_emb_dim: int = 64,
        pair_features: bool = False,
        n_rbf: int = 16,
        rbf_min_a: float = 0.0,
        rbf_max_a: float = 30.0,
        rbf_sigma_a: float = 2.0,
        mlp_block: bool = False,
        mlp_dim: int = 256,
    ):
        super().__init__(
            trunk_dim=trunk_dim, hidden_dim=hidden_dim, score_dim=score_dim,
            n_layers=n_layers, n_heads=n_heads,
            pair_features=pair_features, n_rbf=n_rbf,
            rbf_min_a=rbf_min_a, rbf_max_a=rbf_max_a, rbf_sigma_a=rbf_sigma_a,
            mlp_block=mlp_block, mlp_dim=mlp_dim,
        )
        self.t_emb_dim = t_emb_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def sinusoidal_t_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] float → [B, t_emb_dim]"""
        half = self.t_emb_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self,
        layer_inputs: torch.Tensor,
        t: torch.Tensor,
        coords_nm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            layer_inputs: [B, n_layers, N, trunk_dim]
            t:            [B] float in [0, 1]
            coords_nm:    [B, N, 3] — required iff pair_features=True
        """
        t_emb = self.sinusoidal_t_embedding(t)
        t_cond = self.t_mlp(t_emb)
        r = F.relu(self.W_in(layer_inputs))
        r = r + t_cond[:, None, None, :]
        if self.mlp is not None:
            r = self.mlp(r)
        q = torch.einsum("blnd,lhde->blhne", r, self.W_q)
        k = torch.einsum("blnd,lhde->blhne", r, self.W_k)
        scores = torch.einsum("blhid,blhjd->blhij", q, k)
        scores = scores / (self.score_dim ** 0.5)
        if self.pair_features:
            assert coords_nm is not None
            rbf = self._compute_rbf(coords_nm)
            scores = scores + torch.einsum("bijk,lhk->blhij", rbf, self.W_pair)
        return scores

    def forward_one_layer(
        self,
        layer_input: torch.Tensor,
        layer_idx: int,
        t: torch.Tensor,
        coords_nm: torch.Tensor | None = None,
        precomputed_rbf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One-layer variant with t conditioning. See TopKRouter.forward_one_layer."""
        assert 0 <= layer_idx < self.n_layers, (
            f"layer_idx {layer_idx} out of range [0, {self.n_layers})"
        )
        t_emb = self.sinusoidal_t_embedding(t)
        t_cond = self.t_mlp(t_emb)                                  # [B, hid]
        r = F.relu(self.W_in(layer_input))                          # [B, N, hid]
        r = r + t_cond[:, None, :]                                  # broadcast across N
        if self.mlp is not None:
            r = self.mlp(r)
        Wq_l = self.W_q[layer_idx]
        Wk_l = self.W_k[layer_idx]
        q = torch.einsum("bnd,hde->bhne", r, Wq_l)
        k = torch.einsum("bnd,hde->bhne", r, Wk_l)
        scores = torch.einsum("bhid,bhjd->bhij", q, k)
        scores = scores / (self.score_dim ** 0.5)
        if self.pair_features:
            if precomputed_rbf is None:
                assert coords_nm is not None
                rbf = self._compute_rbf(coords_nm)
            else:
                rbf = precomputed_rbf
            Wp_l = self.W_pair[layer_idx]
            scores = scores + torch.einsum("bijk,hk->bhij", rbf, Wp_l)
        return scores
