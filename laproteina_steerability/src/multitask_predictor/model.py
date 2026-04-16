"""Per-residue transformer with FiLM t-conditioning for multi-task property prediction."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, pre-computed up to max_len."""

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
        # x: [B, L, d_model]
        return x + self.pe[:, :x.size(1)]


class FiLMConditioner(nn.Module):
    """Produce (scale, shift) for a single transformer layer from t embedding."""

    def __init__(self, t_embed_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(t_embed_dim, 2 * d_model)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # t_emb: [B, t_embed_dim]
        out = self.proj(t_emb)  # [B, 2*d_model]
        scale, shift = out.chunk(2, dim=-1)  # each [B, d_model]
        scale = scale + 1.0  # center around identity
        return scale, shift


class FiLMTransformerLayer(nn.Module):
    """Pre-norm transformer layer with FiLM conditioning on the attention sub-layer."""

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
    ) -> torch.Tensor:
        # x: [B, L, d_model], t_emb: [B, t_embed_dim], key_padding_mask: [B, L] (True=valid)
        B, L, D = x.shape

        # FiLM on pre-norm
        normed = self.norm1(x)
        scale, shift = self.film(t_emb)  # [B, D] each
        normed = scale.unsqueeze(1) * normed + shift.unsqueeze(1)

        # Multi-head self-attention
        qkv = self.qkv(normed).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Build attention mask for SDPA: True means IGNORE (opposite of our mask)
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: [B, L], True=valid → invert for SDPA
            attn_mask = ~key_padding_mask  # [B, L], True=pad
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_mask = attn_mask.expand(B, self.n_heads, L, L)
            # SDPA expects float mask with -inf for masked positions, or bool with True=masked
            # Using the boolean convention isn't directly supported for all backends;
            # use float mask for compatibility
            attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill_(attn_mask, float("-inf"))

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )  # [B, H, L, head_dim]

        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        attn_out = self.out_proj(attn_out)
        x = x + self.resid_dropout(attn_out)

        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        return x


class TimeEmbedding(nn.Module):
    """Embed scalar t using sinusoidal features + MLP."""

    def __init__(self, d_model: int, n_freq: int = 32):
        super().__init__()
        self.n_freq = n_freq
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freq, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        freqs = torch.exp(
            torch.arange(self.n_freq, device=t.device, dtype=t.dtype)
            * (-math.log(10000.0) / self.n_freq)
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)  # [B, n_freq]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, 2*n_freq]
        return self.mlp(emb)  # [B, d_model]


class PropertyTransformer(nn.Module):
    """Per-residue transformer with FiLM t-conditioning for 13-property prediction.

    Architecture:
        Input: latents [B, L, 8], mask [B, L], t [B]
        → Linear(8 → d_model)
        → Sinusoidal positional encoding
        → 3 FiLM-conditioned transformer layers
        → Masked mean pooling → [B, d_model]
        → Linear(d_model → 13)
    """

    def __init__(
        self,
        latent_dim: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        n_properties: int = 13,
        max_len: int = 1024,
    ):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.t_embed = TimeEmbedding(d_model)

        self.layers = nn.ModuleList([
            FiLMTransformerLayer(d_model, n_heads, ffn_expansion, dropout)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, n_properties)

    def forward(
        self,
        latents: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            latents: [B, L, 8] per-residue latent vectors
            mask: [B, L] bool, True = valid residue
            t: [B] float, conditioning time (1.0 for clean)

        Returns:
            [B, 13] predicted properties (z-scored scale during training)
        """
        x = self.input_proj(latents)       # [B, L, d_model]
        x = self.pos_enc(x)
        t_emb = self.t_embed(t)            # [B, d_model]

        for layer in self.layers:
            x = layer(x, t_emb, key_padding_mask=mask)

        x = self.output_norm(x)

        # Masked mean pooling
        mask_f = mask.unsqueeze(-1).float()  # [B, L, 1]
        x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)  # [B, d_model]

        return self.output_head(x)  # [B, 13]
