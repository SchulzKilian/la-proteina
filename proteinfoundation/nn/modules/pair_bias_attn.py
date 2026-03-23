# MIT License

# Copyright (c) 2022 MattMcPartlon

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, einsum, nn

from proteinfoundation.nn.modules.adaptive_ln_scale import (
    AdaptiveLayerNorm,
    AdaptiveOutputScale,
)


def exists(val) -> bool:
    """returns whether val is not none"""
    return val is not None


def default(x, y):
    """returns x if it exists, otherwise y"""
    return x if exists(x) else y


max_neg_value = lambda x: torch.finfo(x.dtype).min


class PairBiasAttention(nn.Module):
    """
    Scalar Feature masked attention with pair bias and gating.
    Code modified from
    https://github.com/MattMcPartlon/protein-docking/blob/main/protein_learning/network/modules/node_block.py
    """

    def __init__(
        self,
        node_dim: int,
        dim_head: int,
        heads: int,
        bias: bool,
        dim_out: int,
        qkln: bool,
        pair_dim: Optional[int] = None,
        **kawrgs,  # noqa
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads, self.scale = heads, dim_head**-0.5
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)
        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim))
        self.node_norm = nn.LayerNorm(node_dim)
        self.q_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        if exists(pair_dim):
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            self.to_bias, self.pair_norm = None, None

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Optional[Tensor],
        mask: Optional[Tensor],
        neighbor_idx: Optional[Tensor] = None,
    ) -> Tensor:
        """Multi-head scalar Attention Layer

        :param node_feats: scalar features of shape (b,n,d_s)
        :param pair_feats: pair features of shape (b,n,n,d_e) [dense] or (b,n,K,d_e) [sparse]
        :param mask: (b,n,n) boolean pair mask [dense] or (b,n) sequence mask [sparse]
        :param neighbor_idx: (b,n,K) int tensor for sparse attention; None → dense
        :return:
        """
        assert exists(self.to_bias) or not exists(pair_feats)
        node_feats, h = self.node_norm(node_feats), self.heads
        pair_feats = self.pair_norm(pair_feats) if exists(pair_feats) else None
        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        g = self.to_g(node_feats)
        # pair bias: works for both [b,n,n,h] dense and [b,n,K,h] sparse via "b ... h -> b h ..."
        b = (
            rearrange(self.to_bias(pair_feats), "b ... h -> b h ...")
            if exists(pair_feats)
            else 0
        )
        q, k, v, g = map(
            lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=h), (q, k, v, g)
        )
        if neighbor_idx is not None:
            attn_feats = self._attn_sparse(q, k, v, b, neighbor_idx, mask)
        else:
            attn_feats = self._attn(q, k, v, b, mask)
        attn_feats = rearrange(
            torch.sigmoid(g) * attn_feats, "b h n d -> b n (h d)", h=h
        )
        return self.to_out_node(attn_feats)

    def _attn(self, q, k, v, b, mask: Optional[Tensor]) -> Tensor:
        """Dense attention update."""
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, "b i j -> b () i j")
            sim = sim.masked_fill(~mask, max_neg_value(sim))
        attn = torch.softmax(sim + b, dim=-1)
        return einsum("b h i j, b h j d -> b h i d", attn, v)

    def _attn_sparse(
        self,
        q: Tensor,            # [b, h, n, d]
        k: Tensor,            # [b, h, n, d]
        v: Tensor,            # [b, h, n, d]
        b: Tensor,            # [b, h, n, K]  pair bias (already projected + rearranged)
        neighbor_idx: Tensor, # [b, n, K]
        seq_mask: Optional[Tensor],  # [b, n] residue validity
    ) -> Tensor:
        """Sparse attention: each query attends only to its K neighbors."""
        B, H, N, D = q.shape
        K = neighbor_idx.shape[-1]
        BH = B * H

        # Flatten batch+head dims for efficient gather
        k_bh = k.reshape(BH, N, D)
        v_bh = v.reshape(BH, N, D)

        # Expand neighbor_idx over heads: [b, n, K] → [BH, n, K]
        idx_bh = neighbor_idx.unsqueeze(1).expand(B, H, N, K).reshape(BH, N, K)

        # Gather K neighbors: [BH, N*K] → expand dim → gather → [BH, N, K, D]
        idx_flat   = idx_bh.reshape(BH, N * K)
        idx_flat_d = idx_flat.unsqueeze(-1).expand(BH, N * K, D)
        k_sparse = k_bh.gather(1, idx_flat_d).reshape(BH, N, K, D)
        v_sparse = v_bh.gather(1, idx_flat_d).reshape(BH, N, K, D)

        # Attention scores via einsum (avoids [BH,N,K,D] intermediate in product)
        q_bh = q.reshape(BH, N, D)
        sim = torch.einsum("bnd,bnkd->bnk", q_bh, k_sparse) * self.scale  # [BH, n, K]
        sim = sim.reshape(B, H, N, K)

        # Mask invalid neighbors
        if exists(seq_mask):
            B_idx = torch.arange(B, device=seq_mask.device).view(B, 1, 1).expand(B, N, K)
            nbr_valid = seq_mask[B_idx, neighbor_idx]               # [b, n, K]
            i_valid   = seq_mask[:, :, None].expand(B, N, K)        # [b, n, K]
            attn_mask = nbr_valid & i_valid                          # [b, n, K]
            sim = sim.masked_fill(~attn_mask.unsqueeze(1), max_neg_value(sim))

        attn = torch.softmax(sim + b, dim=-1)  # [b, h, n, K]

        # Aggregate values
        attn_bh = attn.reshape(BH, N, K)
        out = torch.einsum("bnk,bnkd->bnd", attn_bh, v_sparse)     # [BH, n, D]
        return out.reshape(B, H, N, D)


class MultiHeadBiasedAttentionADALN_MM(torch.nn.Module):
    """Pair biased multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(self, dim_token, dim_pair, nheads, dim_cond, use_qkln):
        super().__init__()
        dim_head = int(dim_token // nheads)
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = PairBiasAttention(
            node_dim=dim_token,
            dim_head=dim_head,
            heads=nheads,
            bias=True,
            dim_out=dim_token,
            qkln=use_qkln,
            pair_dim=dim_pair,
        )
        self.scale_output = AdaptiveOutputScale(dim=dim_token, dim_cond=dim_cond)

    def forward(self, x, pair_rep, cond, mask, neighbor_idx=None):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            pair_rep: Pair representation, shape [b, n, n, dim_pair] (dense) or [b, n, K, dim_pair] (sparse)
            mask: Binary mask, shape [b, n]
            neighbor_idx: optional [b, n, K] for sparse attention

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        if neighbor_idx is not None:
            x = self.adaln(x, cond, mask)
            # sparse: pass sequence mask [b, n]; PairBiasAttention handles neighbor masking
            x = self.mha(node_feats=x, pair_feats=pair_rep, mask=mask, neighbor_idx=neighbor_idx)
        else:
            pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]  — original order
            x = self.adaln(x, cond, mask)
            x = self.mha(node_feats=x, pair_feats=pair_rep, mask=pair_mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]
