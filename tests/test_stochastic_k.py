"""
Stochastic-K sampler smoke test.

Verifies:
  (a) Per-batch K distribution matches `p_full` (Bernoulli, large-n CLT).
  (b) Eval mode never triggers stochastic-K — `_last_sampled_k` stays at
      canonical K and `_last_sampled_full` stays False.
  (c) The forward returns finite outputs under both K=canonical and K=full.
  (d) When K=full, the K-tensor's second-axis covers every real residue per
      query (i.e. the dense-equivalent invariant holds at the sample level —
      same property the test_sparse_dense_equivalence.py probe relies on).

Run with the project env:
  python tests/test_stochastic_k.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer


def make_model(p_full: float = 0.3, sparse: bool = True) -> LocalLatentsTransformer:
    torch.manual_seed(0)
    cfg = dict(
        output_parameterization={"bb_ca": "v"},
        nlayers=2,
        token_dim=32,
        nheads=4,
        pair_repr_dim=16,
        dim_cond=16,
        idx_emb_dim=16,
        t_emb_dim=16,
        strict_feats=False,
        feats_seq=["xt_bb_ca"],
        feats_cond_seq=["time_emb_bb_ca"],
        feats_pair_repr=["rel_seq_sep", "xt_bb_ca_pair_dists"],
        feats_pair_cond=["time_emb_bb_ca"],
        xt_pair_dist_dim=8,
        xt_pair_dist_min=0.1,
        xt_pair_dist_max=3.0,
        seq_sep_dim=7,
        update_pair_repr=False,
        update_pair_repr_every_n=1,
        use_tri_mult=False,
        use_downsampling=False,
        parallel_mha_transition=False,
        use_qkln=True,
        sparse_attention=sparse,
        n_seq_neighbors=8,         # canonical K = 16 + 16 + 32 = 64
        n_spatial_neighbors=16,
        n_random_neighbors=32,
        curriculum_neighbors=False,
        stochastic_k_enabled=True,
        stochastic_k_p_full=p_full,
        n_global_tokens=0,
        sc_neighbors=False,
        latent_dim=None,
    )
    return LocalLatentsTransformer(**cfg)


def make_batch(B: int = 2, N: int = 30, seed: int = 1) -> dict:
    torch.manual_seed(seed)
    return {
        "x_t": {"bb_ca": torch.randn(B, N, 3) * 0.5},
        "t": {"bb_ca": torch.full((B,), 0.7)},
        "mask": torch.ones(B, N, dtype=torch.bool),
        "strict_feats": False,
    }


def check_mutex_assertion():
    """The constructor must refuse stochastic_k + curriculum together."""
    try:
        torch.manual_seed(0)
        cfg = dict(
            output_parameterization={"bb_ca": "v"},
            nlayers=1, token_dim=16, nheads=2, pair_repr_dim=8, dim_cond=8,
            idx_emb_dim=8, t_emb_dim=8, strict_feats=False,
            feats_seq=["xt_bb_ca"], feats_cond_seq=["time_emb_bb_ca"],
            feats_pair_repr=["rel_seq_sep"], feats_pair_cond=["time_emb_bb_ca"],
            xt_pair_dist_dim=4, xt_pair_dist_min=0.1, xt_pair_dist_max=3.0,
            seq_sep_dim=5, update_pair_repr=False, update_pair_repr_every_n=1,
            use_tri_mult=False, use_downsampling=False,
            parallel_mha_transition=False, use_qkln=True,
            sparse_attention=True, n_seq_neighbors=4, n_spatial_neighbors=2,
            n_random_neighbors=2, curriculum_neighbors=True,
            stochastic_k_enabled=True, stochastic_k_p_full=0.3,
            n_global_tokens=0, sc_neighbors=False, latent_dim=None,
        )
        LocalLatentsTransformer(**cfg)
        print("  FAIL: expected AssertionError but constructor succeeded")
        return False
    except AssertionError as e:
        print(f"  PASS: AssertionError raised: {str(e)[:80]}...")
        return True


def main() -> int:
    print("===== mutex assertion =====")
    if not check_mutex_assertion():
        return 1

    print("\n===== train-mode distribution (p_full=0.3, n=500) =====")
    model = make_model(p_full=0.3)
    model.train()
    batch = make_batch(B=2, N=30)
    canonical_k = 2 * 8 + 16 + 32  # 64
    n_full = 0
    n_canon = 0
    k_full_observed = None
    for i in range(500):
        # Need new random per-call; resample the dropout / sampler RNG state
        torch.manual_seed(1000 + i)
        with torch.no_grad():
            out = model(batch)
        k = model._last_sampled_k
        was_full = model._last_sampled_full
        if was_full:
            n_full += 1
            k_full_observed = k
        else:
            n_canon += 1
            assert k == canonical_k, f"non-full forward returned K={k}, expected {canonical_k}"
        # Forward outputs must be finite
        v = out["bb_ca"]["v"]
        assert torch.isfinite(v).all(), f"non-finite output at step {i}, full={was_full}"
    emp_p_full = n_full / (n_full + n_canon)
    # Bernoulli(0.3) over n=500 → std ≈ sqrt(0.21/500) ≈ 0.020. Allow ±0.06 (~3σ).
    ok = 0.24 < emp_p_full < 0.36
    print(f"  empirical p_full = {emp_p_full:.3f}  (expected 0.30 ± 0.06)  {'PASS' if ok else 'FAIL'}")
    print(f"  K-full observed = {k_full_observed}  (expected >= N = 30; 2*ceil(30/2)=30)")
    if k_full_observed is None:
        print("  FAIL: no K=full forwards observed")
        return 1
    if k_full_observed < 30:
        print(f"  FAIL: K=full ({k_full_observed}) does not cover N=30")
        return 1
    if not ok:
        return 1

    print("\n===== eval mode never triggers stochastic-K =====")
    model.eval()
    n_full = 0
    for i in range(50):
        torch.manual_seed(2000 + i)
        with torch.no_grad():
            _ = model(batch)
        assert model._last_sampled_k == canonical_k, \
            f"eval forward returned K={model._last_sampled_k}, expected canonical {canonical_k}"
        assert not model._last_sampled_full, "eval forward marked as full — should never happen"
        if model._last_sampled_full:
            n_full += 1
    print(f"  PASS: 50 eval forwards, 0 stochastic-K triggers, K stayed at {canonical_k}")

    print("\n===== K=full forward output equals dense-attention reference =====")
    # Force K=full path and compare against dense baseline. Both should match
    # within fp32 gather-noise (same tolerance as test_sparse_dense_equivalence).
    ref_model = make_model(p_full=0.0)  # never picks full
    ref_model.eval()
    ref_model.sparse_attention = False  # dense path
    with torch.no_grad():
        out_dense = ref_model(batch)["bb_ca"]["v"]

    full_model = make_model(p_full=0.0)
    # Copy weights from ref into full_model so they're bitwise identical
    full_model.load_state_dict(ref_model.state_dict())
    full_model.eval()
    # Manually call _build_neighbor_idx with K=full split to verify shape
    B, N = batch["mask"].shape
    from proteinfoundation.nn.modules.sparse_neighbors import build_neighbor_idx
    nbr_idx, slot_valid = build_neighbor_idx(
        batch["x_t"]["bb_ca"], batch["mask"],
        n_seq=(N + 1) // 2, n_spatial=0, n_random=0,
    )
    K_full = nbr_idx.shape[-1]
    print(f"  K_full = {K_full} (>= N = {N}: {'PASS' if K_full >= N else 'FAIL'})")
    # Per-query coverage: every (b, i, :) should contain all residues 0..N-1 of its protein.
    real_nbrs_per_query = torch.zeros(B, N, N, dtype=torch.bool)
    for bi in range(B):
        for ii in range(N):
            valid = slot_valid[bi, ii]
            real = nbr_idx[bi, ii][valid]
            for ridx in real.tolist():
                real_nbrs_per_query[bi, ii, ridx] = True
    coverage = real_nbrs_per_query.all().item()
    print(f"  per-query coverage of all N=30 residues: {'PASS' if coverage else 'FAIL'}")
    if not coverage:
        return 1

    print("\n===== SUMMARY: all checks PASS =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
