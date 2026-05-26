"""Move-2 router-supplied sparse attention.

Same QKV / pair-bias / output-gating structure as PairBiasAttention, but the
attention's softmax is restricted to a per-(batch, head, query) K-set of K
keys supplied by an external module (e.g. a frozen TopKRouter). All other
keys are masked to -inf before the softmax — the standard mask-then-softmax
renormalization pattern.

Key invariants vs canonical dense:
  - Q/K/V projections, output gating, AdaLN/AdaLN-Zero, pair bias projection
    are bit-identical to PairBiasAttention.
  - The dense pair representation [B, N, N, pair_dim] is built normally and
    the bias [B, H, N, N] is gathered per-(batch, head, query) onto the K-set.
    No special pair_repr_builder path required.
  - Per-head K-set: neighbor_idx shape [B, H, N, K] (unlike the SALAD-style
    sparse path which used [B, N, K] — same K-set across heads).
  - Self-inclusion: enforced by the caller (we don't manipulate the K-set
    here; the trunk wiring prepends self and asks the router for top-K-1
    with self masked to -inf).
  - K=N degenerate case: same code path. Padding slots in the K-set are
    handled via the optional slot_valid argument (False entries are masked
    out before softmax). When N ≤ K, the trunk wiring pads with self-index
    and sets slot_valid=False for the pads.
"""
from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, einsum

from proteinfoundation.nn.modules.adaptive_ln_scale import (
    AdaptiveLayerNorm,
    AdaptiveOutputScale,
)
from proteinfoundation.nn.modules.pair_bias_attn import (
    PairBiasAttention,
    max_neg_value,
)


class PairBiasAttentionSparse(PairBiasAttention):
    """Router-K-set variant of PairBiasAttention.

    Constructor and parameter layout are identical to PairBiasAttention so
    state_dicts are interchangeable. Only the attention computation changes.
    """

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Optional[Tensor],
        mask: Optional[Tensor],
        router_neighbor_idx: Tensor,
        slot_valid: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with router-supplied per-(head, query) K-set.

        Args:
            node_feats: [B, N, node_dim] sequence tokens.
            pair_feats: [B, N, N, pair_dim] DENSE pair representation.
            mask: [B, N] bool sequence mask (True = real query/key).
            router_neighbor_idx: [B, H, N, K] long — per-(batch, head, query)
                key indices into [0, N).
            slot_valid: [B, H, N, K] bool — True for real K-set entries,
                False for padding slots (only relevant when N < K).
        """
        assert pair_feats is not None and self.to_bias is not None, (
            "PairBiasAttentionSparse requires pair_feats (canonical config "
            "passes them; the K-set restricts attention but pair bias is "
            "still per-(head, query, key)."
        )
        node_feats = self.node_norm(node_feats)
        pair_feats = self.pair_norm(pair_feats)
        h = self.heads

        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        g = self.to_g(node_feats)

        # Dense bias [B, H, N, N] — same as canonical, gathered per-head below.
        b_dense = rearrange(self.to_bias(pair_feats), "b i j h -> b h i j")
        q, k, v, g = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v, g)
        )

        attn_feats = self._attn_router_sparse(
            q, k, v, b_dense, router_neighbor_idx, mask, slot_valid
        )

        attn_feats = rearrange(
            torch.sigmoid(g) * attn_feats, "b h n d -> b n (h d)", h=h
        )
        return self.to_out_node(attn_feats)

    def _attn_router_sparse(
        self,
        q: Tensor,                  # [B, H, N, D]
        k: Tensor,                  # [B, H, N, D]
        v: Tensor,                  # [B, H, N, D]
        b_dense: Tensor,            # [B, H, N, N] dense pair bias
        router_neighbor_idx: Tensor,  # [B, H, N, K]
        seq_mask: Optional[Tensor],   # [B, N]
        slot_valid: Optional[Tensor],  # [B, H, N, K] bool or None
    ) -> Tensor:
        """Sparse attention: each (head, query) attends only to its K router-chosen keys."""
        B, H, N, D = q.shape
        K = router_neighbor_idx.shape[-1]
        assert router_neighbor_idx.shape == (B, H, N, K), (
            f"router_neighbor_idx shape mismatch: expected ({B},{H},{N},{K}), "
            f"got {tuple(router_neighbor_idx.shape)}"
        )

        # ---- Gather K_sparse, V_sparse: [B, H, N, K, D] ----
        # Flatten (B, H) for efficient gather along the N (key) axis.
        BH = B * H
        k_bh = k.reshape(BH, N, D)
        v_bh = v.reshape(BH, N, D)
        idx_bh = router_neighbor_idx.reshape(BH, N, K)  # [BH, N, K]
        idx_flat = idx_bh.reshape(BH, N * K)
        idx_flat_d = idx_flat.unsqueeze(-1).expand(BH, N * K, D)
        k_sparse = k_bh.gather(1, idx_flat_d).reshape(BH, N, K, D)
        v_sparse = v_bh.gather(1, idx_flat_d).reshape(BH, N, K, D)

        # ---- QK^T scaled: [B, H, N, K] ----
        q_bh = q.reshape(BH, N, D)
        sim = torch.einsum("bnd,bnkd->bnk", q_bh, k_sparse) * self.scale
        sim = sim.reshape(B, H, N, K)

        # ---- Pair bias gather: b_dense[b, h, i, idx[b,h,i,k]] -> [B, H, N, K] ----
        b_gathered = b_dense.gather(-1, router_neighbor_idx)  # [B, H, N, K]
        # Note: explicit `del` on k_sparse / b_dense doesn't free memory here —
        # the autograd graph keeps references for backward. To actually reduce
        # peak activation memory, wrap this method with torch.utils.checkpoint
        # (deferred — needs a separate change since it affects the forward graph).

        # ---- Validity mask ----
        # Any K-set position pointing at a padded key is invalid; any query
        # that is itself padded is irrelevant (we still compute it but the
        # downstream mask zeros its contribution).
        if seq_mask is not None:
            # nbr_valid[b, h, i, k] = seq_mask[b, idx[b, h, i, k]]
            seq_mask_idx = seq_mask.view(B, 1, 1, N).expand(B, H, N, N)
            nbr_valid = seq_mask_idx.gather(-1, router_neighbor_idx)  # [B, H, N, K]
            q_valid = seq_mask.view(B, 1, N, 1).expand(B, H, N, K)
            attn_mask = nbr_valid & q_valid
            if slot_valid is not None:
                attn_mask = attn_mask & slot_valid
            sim = sim.masked_fill(~attn_mask, max_neg_value(sim))
        elif slot_valid is not None:
            sim = sim.masked_fill(~slot_valid, max_neg_value(sim))

        # ---- Softmax over kept K keys, then weighted sum of V_sparse ----
        attn = torch.softmax(sim + b_gathered, dim=-1).nan_to_num(0.0)  # [B, H, N, K]
        attn_bh = attn.reshape(BH, N, K)
        out = torch.einsum("bnk,bnkd->bnd", attn_bh, v_sparse)  # [BH, N, D]
        return out.reshape(B, H, N, D)


class MultiHeadBiasedAttentionADALN_MM_RouterSparse(torch.nn.Module):
    """AdaLN-Zero wrapped sparse MHA — mirrors MultiHeadBiasedAttentionADALN_MM
    but uses PairBiasAttentionSparse and consumes a per-head K-set."""

    def __init__(self, dim_token, dim_pair, nheads, dim_cond, use_qkln):
        super().__init__()
        dim_head = int(dim_token // nheads)
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = PairBiasAttentionSparse(
            node_dim=dim_token,
            dim_head=dim_head,
            heads=nheads,
            bias=True,
            dim_out=dim_token,
            qkln=use_qkln,
            pair_dim=dim_pair,
        )
        self.scale_output = AdaptiveOutputScale(dim=dim_token, dim_cond=dim_cond)

    def forward(self, x, pair_rep, cond, mask, router_neighbor_idx, slot_valid=None):
        x = self.adaln(x, cond, mask)
        x = self.mha(
            node_feats=x,
            pair_feats=pair_rep,
            mask=mask,
            router_neighbor_idx=router_neighbor_idx,
            slot_valid=slot_valid,
        )
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class MultiheadAttnAndTransitionRouterSparse(torch.nn.Module):
    """Single transformer block: router-sparse MHA + AdaLN transition."""

    def __init__(
        self,
        dim_token,
        dim_pair,
        nheads,
        dim_cond,
        use_qkln,
        expansion_factor=4,
    ):
        super().__init__()
        from proteinfoundation.nn.modules.seq_transition_af3 import TransitionADALN

        self.mhba = MultiHeadBiasedAttentionADALN_MM_RouterSparse(
            dim_token=dim_token,
            dim_pair=dim_pair,
            nheads=nheads,
            dim_cond=dim_cond,
            use_qkln=use_qkln,
        )
        self.transition = TransitionADALN(
            dim=dim_token, dim_cond=dim_cond, expansion_factor=expansion_factor
        )

    def _apply_mha(self, x, pair_rep, cond, mask, router_neighbor_idx, slot_valid):
        x_attn = self.mhba(
            x, pair_rep, cond, mask,
            router_neighbor_idx=router_neighbor_idx,
            slot_valid=slot_valid,
        )
        # canonical residual_mha=True
        x_attn = x_attn + x
        return x_attn * mask[..., None]

    def _apply_transition(self, x, cond, mask):
        x_tr = self.transition(x, cond, mask)
        # canonical residual_transition=True
        x_tr = x_tr + x
        return x_tr * mask[..., None]

    def forward(self, x, pair_rep, cond, mask, router_neighbor_idx, slot_valid=None):
        x = x * mask[..., None]
        x = self._apply_mha(x, pair_rep, cond, mask, router_neighbor_idx, slot_valid)
        x = self._apply_transition(x, cond, mask)
        return x * mask[..., None]
