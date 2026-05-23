"""CA-conditioned per-residue transformer with FiLM t-conditioning + distance pair-bias.

Identical to `multitask_predictor.model.PropertyTransformer` except:

  * forward() also accepts `coords: [B, L, 3]` (per-residue CA, in nm).
  * a small `CoordPairBias` module turns coords into a per-head additive
    bias on the attention logits. The pair feature is SE(3)-invariant by
    construction (pairwise distances + |i-j| sequence offset), so the
    model has no orientation/translation to learn around.

Everything else (FiLM scale/shift on attention pre-norm, mean-pool, output
head shape) is byte-equivalent to the latent-only predictor.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


# --------------------------------------------------------------------------------------
# Components borrowed unchanged from multitask_predictor.model — re-implemented here so
# this module has no import-time coupling to the latent-only predictor (cleaner ckpt
# loading: each ckpt loads its own model class without any sys.path shuffling).
# --------------------------------------------------------------------------------------


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TimeEmbedding(nn.Module):
    def __init__(self, d_model: int, n_freq: int = 32):
        super().__init__()
        self.n_freq = n_freq
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freq, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freqs = torch.exp(
            torch.arange(self.n_freq, device=t.device, dtype=t.dtype)
            * (-math.log(10000.0) / self.n_freq)
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class FiLMConditioner(nn.Module):
    def __init__(self, t_embed_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(t_embed_dim, 2 * d_model)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.proj(t_emb)
        scale, shift = out.chunk(2, dim=-1)
        scale = scale + 1.0
        return scale, shift


# --------------------------------------------------------------------------------------
# Coord pair-bias.
#
# Pair feature for residues (i, j) is the concatenation of
#   * RBF expansion of d_ij = ||ca_i - ca_j|| in nm   (32 kernels, centers 0..8 nm)
#   * embedding of clamp(i - j, -32, 32)              (65-vocab, 32-dim)
# A single Linear maps the 64-dim feature to n_heads → per-head additive bias on
# the attention logits. Recomputed once per forward, broadcast across all layers.
# Mask handling: invalid (i, j) entries are zeroed; the existing key-padding mask
# in the attention sub-layer already adds -inf to padded keys, so the pair bias
# leaking into a padded slot can't survive softmax.
# --------------------------------------------------------------------------------------


class CoordPairBias(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_rbf: int = 32,
        rbf_max_nm: float = 8.0,
        relpos_clamp: int = 32,
        relpos_dim: int = 32,
    ):
        super().__init__()
        self.n_rbf = n_rbf
        self.relpos_clamp = relpos_clamp

        # RBF centers (in nm). Kept as a buffer so they move with .to(device).
        centers = torch.linspace(0.0, rbf_max_nm, n_rbf, dtype=torch.float32)
        self.register_buffer("rbf_centers", centers)  # [n_rbf]
        # Width: spacing between adjacent centers (works for n_rbf >= 2)
        if n_rbf < 2:
            raise ValueError(f"n_rbf must be >= 2, got {n_rbf}")
        self.rbf_inv_var = 1.0 / max((rbf_max_nm / (n_rbf - 1)) ** 2, 1e-8)

        self.relpos_embed = nn.Embedding(2 * relpos_clamp + 1, relpos_dim)
        self.bias_proj = nn.Linear(n_rbf + relpos_dim, n_heads)

    def forward(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [B, L, 3] CA coordinates in nm
            mask:   [B, L] bool, True = valid residue
        Returns:
            pair_bias: [B, n_heads, L, L] per-head additive bias on attention logits.
        """
        B, L, _ = coords.shape

        # Pairwise distance [B, L, L]
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)        # [B, L, L, 3]
        d = diff.norm(dim=-1)                                   # [B, L, L]

        # RBF expansion [B, L, L, n_rbf]
        d_centered = d.unsqueeze(-1) - self.rbf_centers          # [B, L, L, n_rbf]
        rbf = torch.exp(-self.rbf_inv_var * d_centered ** 2)

        # Sequence rel-pos [L, L] -> embed
        idx = torch.arange(L, device=coords.device)
        relpos = idx.unsqueeze(0) - idx.unsqueeze(1)              # [L, L]  (j - i; sign retained)
        relpos = relpos.clamp(-self.relpos_clamp, self.relpos_clamp) + self.relpos_clamp
        rp = self.relpos_embed(relpos)                            # [L, L, relpos_dim]
        rp = rp.unsqueeze(0).expand(B, -1, -1, -1)                # [B, L, L, relpos_dim]

        feat = torch.cat([rbf, rp], dim=-1)                       # [B, L, L, n_rbf+relpos_dim]
        bias = self.bias_proj(feat)                               # [B, L, L, n_heads]

        # Zero out invalid pairs so they can't accidentally introduce signal even
        # though the downstream key-padding mask should kill them.
        pair_valid = mask.unsqueeze(2) & mask.unsqueeze(1)         # [B, L, L]
        bias = bias * pair_valid.unsqueeze(-1).to(bias.dtype)

        return bias.permute(0, 3, 1, 2).contiguous()              # [B, H, L, L]


# --------------------------------------------------------------------------------------
# FiLM transformer layer with pair-bias hook.
# --------------------------------------------------------------------------------------


class FiLMTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        ffn_dim = d_model * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.film = FiLMConditioner(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        pair_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        normed = self.norm1(x)
        scale, shift = self.film(t_emb)
        normed = scale.unsqueeze(1) * normed + shift.unsqueeze(1)

        qkv = self.qkv(normed).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Build combined attn mask. SDPA wants additive float mask broadcastable
        # to [B, H, L, L]. We start from the pair_bias (or zeros) and -inf out
        # padded keys on top.
        if pair_bias is not None:
            attn_mask = pair_bias.to(q.dtype)                     # [B, H, L, L]
        else:
            attn_mask = None

        if key_padding_mask is not None:
            pad = ~key_padding_mask                                # [B, L]  True = pad
            pad_full = pad.unsqueeze(1).unsqueeze(2).expand(B, self.n_heads, L, L)
            if attn_mask is None:
                attn_mask = torch.zeros((B, self.n_heads, L, L), dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill(pad_full, float("-inf"))

        # Force the math kernel: flash/efficient backward kernels error with
        # "LSE is not correctly aligned (strideH)" when gradient flows back
        # through `attn_mask` (the coord-channel steering case, where
        # `bb_ca` requires_grad). At L ≤ 800 the math kernel is fast enough.
        with sdpa_kernel([SDPBackend.MATH]):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )

        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        attn_out = self.out_proj(attn_out)
        x = x + self.resid_dropout(attn_out)

        x = x + self.ffn(self.norm2(x))
        return x


# --------------------------------------------------------------------------------------
# Top-level predictor.
# --------------------------------------------------------------------------------------


class PropertyTransformerCoords(nn.Module):
    """Per-residue transformer with FiLM t-conditioning + CA distance pair-bias.

    Architecture:
        Input: latents [B, L, 8], coords [B, L, 3], mask [B, L], t [B]
        → Linear(8 → d_model)
        → Sinusoidal positional encoding
        → CoordPairBias(coords, mask) → [B, H, L, L] (computed once)
        → n_layers × FiLM-conditioned transformer layers with pair_bias
        → LayerNorm + masked mean pool → [B, d_model]
        → Linear(d_model → n_properties)
    """

    def __init__(
        self,
        latent_dim: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        n_properties: int = 14,
        max_len: int = 1024,
        # CoordPairBias hyperparams (kept here so they land in the ckpt config)
        n_rbf: int = 32,
        rbf_max_nm: float = 8.0,
        relpos_clamp: int = 32,
        relpos_dim: int = 32,
    ):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.t_embed = TimeEmbedding(d_model)
        self.pair_bias = CoordPairBias(
            n_heads=n_heads,
            n_rbf=n_rbf,
            rbf_max_nm=rbf_max_nm,
            relpos_clamp=relpos_clamp,
            relpos_dim=relpos_dim,
        )

        self.layers = nn.ModuleList([
            FiLMTransformerLayer(d_model, n_heads, ffn_expansion, dropout)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, n_properties)

    def forward(
        self,
        latents: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            latents: [B, L, 8] per-residue latent vectors
            coords:  [B, L, 3] CA coordinates (nm). Already padded with zeros at
                     masked positions.
            mask:    [B, L] bool, True = valid residue
            t:       [B] float
        Returns:
            [B, n_properties] (z-scored at training time)
        """
        x = self.input_proj(latents)
        x = self.pos_enc(x)
        t_emb = self.t_embed(t)

        pb = self.pair_bias(coords, mask)  # [B, H, L, L]

        for layer in self.layers:
            x = layer(x, t_emb, key_padding_mask=mask, pair_bias=pb)

        x = self.output_norm(x)

        mask_f = mask.unsqueeze(-1).float()
        x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return self.output_head(x)
