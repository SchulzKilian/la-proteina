"""
Bit-identity sanity check for the per-layer sparse/dense routing wiring in
LocalLatentsTransformer. Run BEFORE any sweep cell.

Tests:
  A) all-sparse parity:
     M_global (sparse_attention=True, layer_sparse_mask=None)         forward(x)
     ==
     M_hybrid (sparse_attention=True, layer_sparse_mask=[True]*14)    forward(x)

  B) all-dense parity:
     M_global (sparse_attention=False, layer_sparse_mask=None)        forward(x)
     ==
     M_hybrid (sparse_attention=True,  layer_sparse_mask=[False]*14)  forward(x)

If either max-abs-diff is above the fp tolerance the new wiring has a bug
and the sweep is invalid.

Uses random weights — the equivalence claim is about the forward path, not
the weights. Same nn config kwargs as the dense baseline (160M CA-only).
"""
import sys
import torch

sys.path.insert(0, "/home/ks2218/la-proteina")
from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer


NN_KWARGS_DENSE = dict(
    name="local_latents_transformer",
    output_parameterization={"bb_ca": "v"},
    nlayers=14,
    token_dim=768,
    nheads=12,
    parallel_mha_transition=False,
    strict_feats=False,
    feats_seq=["xt_bb_ca", "x_sc_bb_ca",
               "optional_ca_coors_nm_seq_feat", "optional_res_type_seq_feat"],
    feats_cond_seq=["time_emb_bb_ca"],
    dim_cond=256,
    idx_emb_dim=256,
    t_emb_dim=256,
    feats_pair_repr=["rel_seq_sep", "xt_bb_ca_pair_dists",
                     "x_sc_bb_ca_pair_dists", "optional_ca_pair_dist"],
    feats_pair_cond=["time_emb_bb_ca"],
    xt_pair_dist_dim=30, xt_pair_dist_min=0.1, xt_pair_dist_max=3,
    x_sc_pair_dist_dim=30, x_sc_pair_dist_min=0.1, x_sc_pair_dist_max=3,
    seq_sep_dim=127,
    pair_repr_dim=256,
    update_pair_repr=False,
    update_pair_repr_every_n=3,
    use_tri_mult=False,
    use_downsampling=False,
    use_qkln=True,
    sparse_attention=False,
    n_seq_neighbors=8,
    n_spatial_neighbors=8,
    n_random_neighbors=16,
)


def make_batch(B, N, device, dtype):
    """Minimal batch dict for forward()."""
    bb_ca = torch.randn(B, N, 3, device=device, dtype=dtype)
    x_sc_bb_ca = torch.randn(B, N, 3, device=device, dtype=dtype)
    t = torch.rand(B, device=device, dtype=dtype) * 0.5 + 0.25  # mid-t
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    return {
        "x_t":  {"bb_ca": bb_ca},
        "x_sc": {"bb_ca": x_sc_bb_ca},
        "t":    {"bb_ca": t},
        "mask": mask,
        "strict_feats": False,
    }


def build_model(sparse_attention, K_config, device, dtype, seed=0):
    """Build a fresh model with the given sparse_attention flag.
    K_config is (n_seq, n_spatial, n_random) and only matters when
    sparse_attention=True; for the dense model it's irrelevant.
    """
    torch.manual_seed(seed)
    kw = dict(NN_KWARGS_DENSE)
    kw["sparse_attention"] = sparse_attention
    if sparse_attention:
        n_seq, n_sp, n_rd = K_config
        kw["n_seq_neighbors"] = n_seq
        kw["n_spatial_neighbors"] = n_sp
        kw["n_random_neighbors"] = n_rd
    model = LocalLatentsTransformer(**kw).to(device=device, dtype=dtype)
    model.eval()
    return model


def forward_and_extract(model, batch):
    """Single forward pass; return the bb_ca v-prediction tensor."""
    with torch.no_grad():
        out = model(batch)
    return out["bb_ca"]["v"]


def max_abs_diff(a, b):
    return (a - b).abs().max().item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # bit-identity check needs fp32, not bf16
    B, N = 2, 64
    nlayers = NN_KWARGS_DENSE["nlayers"]
    # K=64 SALAD-canonical: (8, 16, 32). Matches the sweep's K=64 cell.
    K_config = (8, 16, 32)

    print(f"device: {device}, dtype: {dtype}, B={B}, N={N}, "
          f"K=(2*{K_config[0]}+{K_config[1]}+{K_config[2]})={2*K_config[0]+K_config[1]+K_config[2]}")

    # ---- Test A: all-sparse parity ----
    print("\n[A] all-sparse parity: layer_sparse_mask=[True]*14 vs global sparse")
    M_global_sparse = build_model(sparse_attention=True,  K_config=K_config,
                                  device=device, dtype=dtype, seed=42)
    M_hybrid        = build_model(sparse_attention=True,  K_config=K_config,
                                  device=device, dtype=dtype, seed=42)
    # Same seed → identical weights. Confirm.
    for (n1, p1), (n2, p2) in zip(M_global_sparse.named_parameters(),
                                  M_hybrid.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"weights differ at {n1} — seed didn't match"

    M_hybrid.layer_sparse_mask = [True] * nlayers
    torch.manual_seed(123)
    batch_A = make_batch(B, N, device, dtype)
    y_global = forward_and_extract(M_global_sparse, batch_A)
    torch.manual_seed(123)  # same Gumbel noise in neighbor builder
    batch_A2 = make_batch(B, N, device, dtype)
    y_hybrid = forward_and_extract(M_hybrid, batch_A2)
    diff_A = max_abs_diff(y_global, y_hybrid)
    print(f"  max abs diff: {diff_A:.2e}")
    # Tolerance: gather may cause tiny fp-roundoff reorderings; <1e-5 in fp32 is safe.
    tol_A = 1e-5
    if diff_A > tol_A:
        print(f"  FAIL — exceeds tol {tol_A:.0e}. Hybrid all-sparse != global sparse.")
        sys.exit(1)
    print(f"  PASS (tol {tol_A:.0e})")

    # ---- Test B: all-dense parity ----
    print("\n[B] all-dense parity: layer_sparse_mask=[False]*14 vs global dense")
    M_global_dense = build_model(sparse_attention=False, K_config=K_config,
                                 device=device, dtype=dtype, seed=42)
    # Hybrid model must be built with sparse_attention=True (asserts require it)
    # but its weights are identical to M_global_dense because they share the seed
    # AND sparse_attention adds no parameters (only enables the neighbor-list method).
    M_hybrid2 = build_model(sparse_attention=True, K_config=K_config,
                            device=device, dtype=dtype, seed=42)
    for (n1, p1), (n2, p2) in zip(M_global_dense.named_parameters(),
                                  M_hybrid2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"weights differ at {n1}"

    M_hybrid2.layer_sparse_mask = [False] * nlayers
    torch.manual_seed(123)
    batch_B = make_batch(B, N, device, dtype)
    y_global2 = forward_and_extract(M_global_dense, batch_B)
    torch.manual_seed(123)
    batch_B2 = make_batch(B, N, device, dtype)
    y_hybrid2 = forward_and_extract(M_hybrid2, batch_B2)
    diff_B = max_abs_diff(y_global2, y_hybrid2)
    print(f"  max abs diff: {diff_B:.2e}")
    # All-dense hybrid receives neighbor_idx=None at every layer, exactly like
    # the global-dense path. The dense pair_rep was rebuilt from the same module
    # weights, so the values should match exactly (no fp reordering).
    tol_B = 1e-7  # tighter; no gather in the all-dense path
    if diff_B > tol_B:
        print(f"  FAIL — exceeds tol {tol_B:.0e}. Hybrid all-dense != global dense.")
        sys.exit(1)
    print(f"  PASS (tol {tol_B:.0e})")

    # ---- Smoke test C: hybrid 7-7 split runs without crash ----
    print("\n[C] smoke: layer_sparse_mask=[True]*7+[False]*7 runs end-to-end")
    M_hybrid3 = build_model(sparse_attention=True, K_config=K_config,
                            device=device, dtype=dtype, seed=42)
    M_hybrid3.layer_sparse_mask = [True] * 7 + [False] * 7
    torch.manual_seed(456)
    batch_C = make_batch(B, N, device, dtype)
    y_hybrid3 = forward_and_extract(M_hybrid3, batch_C)
    print(f"  output shape: {tuple(y_hybrid3.shape)}, no NaN: "
          f"{not bool(torch.isnan(y_hybrid3).any())}")

    print("\nAll three checks passed.")


if __name__ == "__main__":
    main()
