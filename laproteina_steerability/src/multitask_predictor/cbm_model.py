"""Concept-bottleneck property predictor.

latent  -->  [ amino-acid identity + backbone/side-chain torsion angles ]  -->  properties
            \________________________ bottleneck ________________________/

Drop-in for ``PropertyTransformer``: ``forward(latents, mask, t) -> [B, n_properties]``
returns the same z-scored property vector, so it plugs into ``SteeringPredictor`` unchanged.
The difference is purely the bottleneck in the middle: the property heads (g2) see ONLY the
predicted AA distribution + torsion angles, never the raw latent. Any gradient into the latent
therefore has to route through the predicted amino-acid / torsion concepts, which are supervised
against the ground truth — closing the latent->property shortcut that gets gradient-hacked.

g1 (latent -> bottleneck) and g2 (bottleneck -> properties) are each the same FiLM-t transformer
trunk as the original predictor; g1 ends in AA + torsion heads, g2 begins from the bottleneck.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SinusoidalPositionalEncoding, FiLMTransformerLayer, TimeEmbedding

# OpenFold restype order: 0..19 standard AAs, 20 = X/unknown -> 21 classes.
N_AA = 21
# OpenFold atom37_to_torsion_angles: [omega, phi, psi, chi1, chi2, chi3, chi4].
N_TORSIONS = 7


class CBMPropertyTransformer(nn.Module):
    def __init__(
        self,
        latent_dim: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        n_properties: int = 14,
        n_aa: int = N_AA,
        n_torsions: int = N_TORSIONS,
        max_len: int = 1024,
    ):
        super().__init__()
        self.n_aa = n_aa
        self.n_torsions = n_torsions

        # ---- g1: latent -> per-residue features -> bottleneck heads ----
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.enc_pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.t_embed = TimeEmbedding(d_model)
        self.enc_layers = nn.ModuleList([
            FiLMTransformerLayer(d_model, n_heads, ffn_expansion, dropout)
            for _ in range(n_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)
        self.aa_head = nn.Linear(d_model, n_aa)
        self.torsion_head = nn.Linear(d_model, n_torsions * 2)  # (sin, cos) per torsion

        # ---- g2: bottleneck -> properties (mirror of PropertyTransformer) ----
        self.bottleneck_proj = nn.Linear(n_aa + n_torsions * 2, d_model)
        self.dec_pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.dec_layers = nn.ModuleList([
            FiLMTransformerLayer(d_model, n_heads, ffn_expansion, dropout)
            for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, n_properties)

    # ------------------------------------------------------------------
    def encode_bottleneck(self, latents, mask, t_emb):
        """latents [B,L,latent_dim] -> aa_logits [B,L,n_aa], torsions [B,L,n_torsions,2]."""
        x = self.input_proj(latents)
        x = self.enc_pos(x)
        for layer in self.enc_layers:
            x = layer(x, t_emb, key_padding_mask=mask)
        x = self.enc_norm(x)
        aa_logits = self.aa_head(x)
        tors = self.torsion_head(x).reshape(x.shape[0], x.shape[1], self.n_torsions, 2)
        tors = F.normalize(tors, dim=-1, eps=1e-6)  # unit (sin, cos)
        return aa_logits, tors

    def decode_properties(self, aa_probs, tors, mask, t_emb):
        """bottleneck (aa_probs [B,L,n_aa], tors [B,L,n_torsions,2]) -> props [B,n_properties]."""
        b = torch.cat([aa_probs, tors.flatten(-2)], dim=-1)  # [B,L,n_aa+2*n_torsions]
        h = self.bottleneck_proj(b)
        h = self.dec_pos(h)
        for layer in self.dec_layers:
            h = layer(h, t_emb, key_padding_mask=mask)
        h = self.output_norm(h)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return self.output_head(pooled)

    def forward(self, latents, mask, t, return_aux: bool = False, teacher=None, hard: bool = False):
        """
        Args:
            latents [B,L,latent_dim], mask [B,L] bool, t [B] float.
            return_aux: also return (aa_logits, torsions) for the bottleneck losses.
            teacher: optional (aa_probs [B,L,n_aa], torsions [B,L,n_torsions,2]) to feed
                     g2 instead of g1's prediction (teacher forcing).
            hard: straight-through one-hot of the AA bottleneck (for hard-decode inference).
        Returns:
            props [B,n_properties] (z-scored), and (aa_logits, torsions) if return_aux.
        """
        t_emb = self.t_embed(t)
        aa_logits, tors = self.encode_bottleneck(latents, mask, t_emb)

        if teacher is not None:
            aa_in, tors_in = teacher
        else:
            aa_in = F.softmax(aa_logits, dim=-1)
            if hard:
                idx = aa_in.argmax(dim=-1, keepdim=True)
                hard_oh = torch.zeros_like(aa_in).scatter_(-1, idx, 1.0)
                aa_in = hard_oh + (aa_in - aa_in.detach())  # straight-through estimator
            tors_in = tors

        props = self.decode_properties(aa_in, tors_in, mask, t_emb)
        if return_aux:
            return props, aa_logits, tors
        return props
