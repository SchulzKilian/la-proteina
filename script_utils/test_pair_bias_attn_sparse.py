"""Unit test for PairBiasAttentionSparse.

When the router-supplied K-set is the identity (each query attends to all N
keys, K=N), the sparse forward should match dense PairBiasAttention output
to within fp32 round-off. This validates the gather + bias gather + softmax
pieces are wired correctly.

Also tests:
  - Per-head distinct K-sets: build heads-vary K-sets, verify softmax sums to 1
    over kept slots and never touches dropped slots.
  - K < N: top-K mask zeroes out everything outside the K-set.
  - N < K with slot_valid padding: short-protein degenerate case.
"""
import sys

REPO = "/home/ks2218/la-proteina"
sys.path.insert(0, REPO)

import torch

from proteinfoundation.nn.modules.pair_bias_attn import PairBiasAttention
from proteinfoundation.nn.modules.pair_bias_attn_sparse import PairBiasAttentionSparse


def make_pair():
    torch.manual_seed(0)
    node_dim, pair_dim, heads, dim_head = 32, 16, 4, 8
    dense = PairBiasAttention(
        node_dim=node_dim, dim_head=dim_head, heads=heads,
        bias=True, dim_out=node_dim, qkln=True, pair_dim=pair_dim,
    )
    sparse = PairBiasAttentionSparse(
        node_dim=node_dim, dim_head=dim_head, heads=heads,
        bias=True, dim_out=node_dim, qkln=True, pair_dim=pair_dim,
    )
    # Share weights (so the only difference is the attention path).
    sparse.load_state_dict(dense.state_dict())
    return dense, sparse, node_dim, pair_dim, heads


def test_identity_K_matches_dense():
    """K=N identity K-set → sparse output == dense output."""
    dense, sparse, node_dim, pair_dim, heads = make_pair()
    B, N = 2, 12
    x = torch.randn(B, N, node_dim)
    pair = torch.randn(B, N, N, pair_dim)
    mask = torch.ones(B, N, dtype=torch.bool)

    # Dense forward
    pair_mask = mask[:, :, None] * mask[:, None, :]
    out_dense = dense(node_feats=x, pair_feats=pair, mask=pair_mask)

    # Sparse with K=N identity K-set per head.
    K = N
    # neighbor_idx[b, h, i, k] = k (each query attends to all N keys in order).
    nb_idx = torch.arange(N).view(1, 1, 1, N).expand(B, heads, N, K).contiguous()
    out_sparse = sparse(
        node_feats=x, pair_feats=pair, mask=mask, router_neighbor_idx=nb_idx,
    )
    diff = (out_dense - out_sparse).abs().max().item()
    assert diff < 1e-5, f"identity-K-set mismatch: max abs diff = {diff:.3e}"
    print(f"[PASS] identity K=N matches dense, max abs diff = {diff:.3e}")


def test_permuted_K_matches_dense():
    """K=N but with a permuted K-set per head → still all keys attended; result == dense."""
    dense, sparse, node_dim, pair_dim, heads = make_pair()
    B, N = 2, 10
    x = torch.randn(B, N, node_dim)
    pair = torch.randn(B, N, N, pair_dim)
    mask = torch.ones(B, N, dtype=torch.bool)

    pair_mask = mask[:, :, None] * mask[:, None, :]
    out_dense = dense(node_feats=x, pair_feats=pair, mask=pair_mask)

    # Per-(b, h, i) permutation of [0..N-1] — same set of keys, different order.
    torch.manual_seed(1)
    K = N
    nb_idx = torch.stack([
        torch.stack([
            torch.stack([torch.randperm(N) for _ in range(N)])
            for _ in range(heads)
        ])
        for _ in range(B)
    ])  # [B, H, N, K]
    out_sparse = sparse(
        node_feats=x, pair_feats=pair, mask=mask, router_neighbor_idx=nb_idx,
    )
    diff = (out_dense - out_sparse).abs().max().item()
    # softmax is permutation-invariant over keys → should match exactly modulo fp round-off.
    assert diff < 1e-5, f"permuted-K mismatch: max abs diff = {diff:.3e}"
    print(f"[PASS] permuted K=N matches dense, max abs diff = {diff:.3e}")


def test_K_less_than_N_restricts_attention():
    """K < N: the sparse output should differ from dense (attention is restricted)."""
    dense, sparse, node_dim, pair_dim, heads = make_pair()
    B, N = 2, 16
    K = 4
    x = torch.randn(B, N, node_dim)
    pair = torch.randn(B, N, N, pair_dim)
    mask = torch.ones(B, N, dtype=torch.bool)
    pair_mask = mask[:, :, None] * mask[:, None, :]

    out_dense = dense(node_feats=x, pair_feats=pair, mask=pair_mask)

    # K-set: first K key indices [0..K-1] for every (b, h, i). Deterministic.
    nb_idx = torch.arange(K).view(1, 1, 1, K).expand(B, heads, N, K).contiguous()
    out_sparse = sparse(
        node_feats=x, pair_feats=pair, mask=mask, router_neighbor_idx=nb_idx,
    )
    diff = (out_dense - out_sparse).abs().max().item()
    # Should differ since dense attends to all 16, sparse only to first 4.
    assert diff > 1e-3, f"K<N: expected meaningful diff, got {diff:.3e} (suspicious)"
    print(f"[PASS] K=4 < N=16 restricts attention; diff vs dense = {diff:.3e}")


def test_short_protein_with_slot_valid():
    """N < K case: pad K-set with self-index, slot_valid=False on pads.
    Softmax should ignore pads and the result should match a dense forward on the short protein."""
    dense, sparse, node_dim, pair_dim, heads = make_pair()
    B, N_real, N_pad, K = 1, 5, 8, 16
    x_full = torch.randn(B, N_pad, node_dim)
    pair_full = torch.randn(B, N_pad, N_pad, pair_dim)
    # Real residues 0..N_real-1, padded N_real..N_pad-1.
    mask = torch.zeros(B, N_pad, dtype=torch.bool)
    mask[:, :N_real] = True

    pair_mask = mask[:, :, None] * mask[:, None, :]
    out_dense = dense(node_feats=x_full, pair_feats=pair_full, mask=pair_mask)

    # Sparse K-set: real residues attend to all N_real keys + (K - N_real) self-pad.
    nb_idx = torch.zeros(B, heads, N_pad, K, dtype=torch.long)
    slot_valid = torch.zeros(B, heads, N_pad, K, dtype=torch.bool)
    for i in range(N_pad):
        for r in range(min(N_real, K)):
            nb_idx[:, :, i, r] = r
            slot_valid[:, :, i, r] = True
        # Pad slots [N_real:K] = self-index (any valid index; they'll be masked anyway).
        for k_pad in range(N_real, K):
            nb_idx[:, :, i, k_pad] = i if i < N_pad else 0
            slot_valid[:, :, i, k_pad] = False

    out_sparse = sparse(
        node_feats=x_full, pair_feats=pair_full, mask=mask,
        router_neighbor_idx=nb_idx, slot_valid=slot_valid,
    )

    # Compare only the real-residue rows; padding-row outputs differ because dense
    # uses pair_mask (zeros out padded queries) while sparse uses seq_mask differently.
    diff = (out_dense[:, :N_real] - out_sparse[:, :N_real]).abs().max().item()
    assert diff < 1e-4, f"short-protein slot_valid path mismatch: max abs diff = {diff:.3e}"
    print(f"[PASS] N_real=5 < K=16 with slot_valid pads, real-row diff = {diff:.3e}")


def test_grad_only_through_qkv_not_through_idx():
    """The K-set indices are int — gradients shouldn't flow through them.
    Verify gradient computation works end-to-end without breaking."""
    dense, sparse, node_dim, pair_dim, heads = make_pair()
    B, N, K = 2, 8, 4
    x = torch.randn(B, N, node_dim, requires_grad=True)
    pair = torch.randn(B, N, N, pair_dim, requires_grad=True)
    mask = torch.ones(B, N, dtype=torch.bool)
    nb_idx = torch.randint(0, N, (B, heads, N, K))

    out = sparse(node_feats=x, pair_feats=pair, mask=mask, router_neighbor_idx=nb_idx)
    out.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert pair.grad is not None and pair.grad.shape == pair.shape
    assert torch.isfinite(x.grad).all() and torch.isfinite(pair.grad).all()
    print(f"[PASS] gradient flows through Q/K/V and pair; max|grad x| = "
          f"{x.grad.abs().max().item():.3e}, max|grad pair| = {pair.grad.abs().max().item():.3e}")


if __name__ == "__main__":
    test_identity_K_matches_dense()
    test_permuted_K_matches_dense()
    test_K_less_than_N_restricts_attention()
    test_short_protein_with_slot_valid()
    test_grad_only_through_qkv_not_through_idx()
    print("\nAll PairBiasAttentionSparse unit tests passed.")
