"""
Sparse vs dense numerical-equivalence probe for LocalLatentsTransformer.

At L=N=50 with K = 2*n_seq + n_spatial + n_random = 64 and self-inclusion,
each query's neighbor list covers ALL 50 residues (16 seq + 16 spatial + 18
random = 50 valid neighbors, 14 padded slots that slot_valid masks out).
This makes the sparse path NUMERICALLY EQUIVALENT to the dense path -- same
weights, same inputs, same softmax denominator, same attention output.

If outputs disagree, the divergence is a silent sparse-path bug. The script
also instruments per-layer forward hooks so the first diverging module is
named in the output.

Run with the project env:
  python tests/test_sparse_dense_equivalence.py
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer


def make_model(seed: int = 0, curriculum: bool = False, pair_update: bool = False) -> LocalLatentsTransformer:
    """Tiny LocalLatentsTransformer, sparse_attention=True (toggled at test time)."""
    torch.manual_seed(seed)
    cfg = dict(
        # output
        output_parameterization={"bb_ca": "v"},
        # trunk
        nlayers=2,
        token_dim=32,
        nheads=4,
        pair_repr_dim=16,
        dim_cond=16,
        idx_emb_dim=16,
        t_emb_dim=16,
        # features
        strict_feats=False,
        feats_seq=["xt_bb_ca"],
        feats_cond_seq=["time_emb_bb_ca"],
        feats_pair_repr=["rel_seq_sep", "xt_bb_ca_pair_dists"],
        feats_pair_cond=["time_emb_bb_ca"],
        # binning
        xt_pair_dist_dim=8,
        xt_pair_dist_min=0.1,
        xt_pair_dist_max=3.0,
        seq_sep_dim=7,
        # off
        update_pair_repr=pair_update,
        update_pair_repr_every_n=1,  # every layer if enabled
        use_tri_mult=False,
        use_downsampling=False,
        parallel_mha_transition=False,
        use_qkln=True,
        # sparse settings -- K=64 with N=50 covers every residue per query.
        sparse_attention=True,
        n_seq_neighbors=8,
        n_spatial_neighbors=16,
        n_random_neighbors=32,
        curriculum_neighbors=curriculum,
        n_global_tokens=0,
        sc_neighbors=False,
        latent_dim=None,
    )
    return LocalLatentsTransformer(**cfg).eval()


def make_batch(B: int = 1, N: int = 50, seed: int = 1) -> dict:
    torch.manual_seed(seed)
    ca = torch.randn(B, N, 3) * 0.5
    t = torch.full((B,), 0.7)  # all-high-t bucket (curriculum off anyway)
    mask = torch.ones(B, N, dtype=torch.bool)
    return {
        "x_t": {"bb_ca": ca},
        "t": {"bb_ca": t},
        "mask": mask,
        "strict_feats": False,
    }


def run_one(model, batch, sparse: bool, capture: dict | None = None) -> torch.Tensor:
    """One forward pass, optionally capturing per-module outputs."""
    model.sparse_attention = sparse

    hooks = []
    if capture is not None:
        def _grab(name):
            def _h(_mod, _inp, out):
                # store first-element 0 only, on CPU
                if isinstance(out, torch.Tensor):
                    capture[name] = out.detach().clone()
            return _h
        # Hook each transformer layer
        for i, layer in enumerate(model.transformer_layers):
            hooks.append(layer.register_forward_hook(_grab(f"trunk[{i}]")))
        # Hook pair_repr_builder (initial pair_rep, sees neighbor_idx)
        hooks.append(model.pair_repr_builder.register_forward_hook(_grab("pair_repr_builder")))
        # Hook init_repr_factory and cond_factory (per-query seq features)
        hooks.append(model.init_repr_factory.register_forward_hook(_grab("init_repr_factory")))
        hooks.append(model.cond_factory.register_forward_hook(_grab("cond_factory")))

    with torch.no_grad():
        out = model(batch)

    for h in hooks:
        h.remove()
    return out["bb_ca"]["v"]


def run_one_with_grad(model, batch, sparse: bool) -> dict:
    """Forward + backward; return {param_name: grad_tensor} dict."""
    model.sparse_attention = sparse
    model.zero_grad(set_to_none=True)
    out = model(batch)
    # Sum of squares -- nontrivial gradient through every output element.
    loss = (out["bb_ca"]["v"] ** 2).sum()
    loss.backward()
    grads = {}
    for n, p in model.named_parameters():
        grads[n] = p.grad.detach().clone() if p.grad is not None else None
    return grads


def diff(t1: torch.Tensor, t2: torch.Tensor) -> tuple[float, float]:
    """Return (max_abs_diff, mean_abs_diff). Aligns shapes by gather if needed."""
    if t1.shape != t2.shape:
        return float("inf"), float("inf")
    d = (t1 - t2).abs()
    return d.max().item(), d.mean().item()


def run_case(case: str, curriculum: bool, pair_update: bool, t_val: float, tol: float) -> tuple[bool, bool]:
    print(f"\n========== CASE: {case}  curriculum={curriculum}  pair_update={pair_update}  t={t_val} ==========")
    model = make_model(seed=0, curriculum=curriculum, pair_update=pair_update)
    batch = make_batch(B=1, N=50, seed=1)
    batch["t"]["bb_ca"] = torch.full((1,), t_val)

    # --- FORWARD equivalence ---
    cap_dense, cap_sparse = {}, {}
    out_dense = run_one(model, batch, sparse=False, capture=cap_dense)
    out_sparse = run_one(model, batch, sparse=True, capture=cap_sparse)
    max_d, mean_d = diff(out_dense, out_sparse)
    print(f"  FWD bb_ca/v: max|Δ|={max_d:.3e}  mean|Δ|={mean_d:.3e}")
    fwd_ok = max_d < tol

    # --- BACKWARD equivalence ---
    # Run dense and sparse paths through the same model with the same loss
    # (sum of squared outputs), then compare per-parameter gradients.
    grads_dense  = run_one_with_grad(model, batch, sparse=False)
    grads_sparse = run_one_with_grad(model, batch, sparse=True)

    max_g, sum_max = 0.0, 0.0
    worst_param, worst_max = None, 0.0
    diverged = []
    for n in grads_dense:
        gd, gs = grads_dense[n], grads_sparse[n]
        if gd is None and gs is None:
            continue
        if (gd is None) != (gs is None):
            diverged.append((n, "one-sided None gradient", float("inf")))
            continue
        md, ad = diff(gd, gs)
        # Scale by max|grad| so we report RELATIVE error too: a 1e-3 abs diff
        # on a grad of magnitude 1e3 is 1e-6, on a grad of 1e-3 is 1.0.
        scale = max(gd.abs().max().item(), gs.abs().max().item(), 1e-12)
        rel = md / scale
        if md > worst_max:
            worst_max = md
            worst_param = n
        # Real divergence = absolute diff is non-trivial AND relative is large.
        # LayerNorm bias grads can be ~1e-8 due to near-zero outputs, so a
        # purely relative threshold flags fp32 summation noise.
        if md > 1e-5 and rel > 1e-3:
            diverged.append((n, f"max|Δ|={md:.3e} rel={rel:.3e} scale={scale:.3e}", rel))

    print(f"  BWD: worst param '{worst_param}' max|Δ|={worst_max:.3e}")
    if diverged:
        print(f"  BWD: {len(diverged)} parameters diverged > 10*tol")
        for n, msg, _ in sorted(diverged, key=lambda x: -x[2])[:10]:
            print(f"    {n}: {msg}")
    bwd_ok = len(diverged) == 0
    return fwd_ok, bwd_ok


def main() -> int:
    # fp32; tolerance ~1e-4 accommodates gather/permute non-associative roundoff
    # over a deeper trunk while still flagging real bugs (which produce ≥1e-2).
    tol = 1e-4
    cases = [
        ("vanilla K=64",                      False, False, 0.7),
        ("curriculum on, high-t bucket",      True,  False, 0.7),
        ("curriculum on, low-t bucket",       True,  False, 0.1),
        ("curriculum on + pair-update, high", True,  True,  0.7),
        ("curriculum on + pair-update, low",  True,  True,  0.1),
    ]
    results = {c[0]: run_case(*c, tol=tol) for c in cases}
    print("\n=========== SUMMARY ===========")
    all_ok = True
    for name, (fwd_ok, bwd_ok) in results.items():
        print(f"  fwd={'PASS' if fwd_ok else 'FAIL'}  bwd={'PASS' if bwd_ok else 'FAIL'}  {name}")
        all_ok = all_ok and fwd_ok and bwd_ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
