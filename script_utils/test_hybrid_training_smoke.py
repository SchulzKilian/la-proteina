"""
Smoke + correctness check for the NATIVE (trained) layer-hybrid forward —
i.e. layer_sparse_mask + layer_K_splits passed as nn-config kwargs and active
during training (not the inference-only generate.py monkeypatch).

Checks:
  1. Construction-from-kwargs: the hybrid model builds from the same kwargs the
     yaml passes (mask + per-layer K_splits), asserts pass.
  2. Deterministic per-layer-K parity (n_random=0 → no stochastic neighbors):
     layer_sparse_mask=[True]*14 + layer_K_splits=[(8,8,0)]*14
       ==
     global sparse with (n_seq=8, n_sp=8, n_rd=0).
     Both deterministic; per-layer rebuild must reproduce the single global
     build → bit-identical (fp32). Validates the per-layer pair_rep rebuild +
     dispatch.
  3. The REAL schedule (front_plateau_then_taper, 7 sparse + 7 dense) runs
     forward + backward; loss is finite; gradients reach the shared
     pair_repr_builder, a sparse layer (0), and a dense layer (13).
  4. Short protein (N < max K) runs without crashing (padding-slot guard).

fp32, CPU-friendly. Random weights — the claims are about the forward/backward
path, not the weights.
"""
import sys

import torch

sys.path.insert(0, "/home/ks2218/la-proteina")
from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer
from script_utils.test_layer_selective_sanity import NN_KWARGS_DENSE, make_batch

MASK_7S7D = [True] * 7 + [False] * 7
KSPLITS_FPT = [  # front_plateau_then_taper [56,56,56,40,32,24,16]
    (8, 16, 24), (8, 16, 24), (8, 16, 24),
    (8, 8, 16), (4, 8, 16), (4, 4, 12), (4, 4, 4),
]


def build(seed, **overrides):
    torch.manual_seed(seed)
    kw = dict(NN_KWARGS_DENSE)
    kw.update(overrides)
    return LocalLatentsTransformer(**kw)


def max_abs_diff(a, b):
    return (a - b).abs().max().item()


def main():
    dtype = torch.float32
    nlayers = NN_KWARGS_DENSE["nlayers"]
    ok = True

    # ---- 1 + 2: deterministic per-layer-K parity ----
    print("[1/4] construction-from-kwargs + deterministic per-layer-K parity")
    B, N = 2, 64
    # Global sparse, deterministic K-set (no random neighbors).
    M_global = build(42, sparse_attention=True,
                     n_seq_neighbors=8, n_spatial_neighbors=8, n_random_neighbors=0
                     ).to(dtype=dtype).eval()
    # Per-layer-K, all layers sparse, uniform deterministic split (8,8,0).
    M_native = build(42, sparse_attention=True,
                     n_seq_neighbors=8, n_spatial_neighbors=8, n_random_neighbors=0,
                     layer_sparse_mask=[True] * nlayers,
                     layer_K_splits=[(8, 8, 0)] * nlayers).to(dtype=dtype).eval()
    assert M_native.layer_sparse_mask == [True] * nlayers
    assert M_native.layer_K_splits == [(8, 8, 0)] * nlayers
    for (n1, p1), (n2, p2) in zip(M_global.named_parameters(),
                                  M_native.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2), f"weight mismatch at {n1}"
    torch.manual_seed(7)
    batch = make_batch(B, N, "cpu", dtype)
    with torch.no_grad():
        torch.manual_seed(7); a = M_global(batch)["bb_ca"]["v"]
        torch.manual_seed(7); b = M_native(batch)["bb_ca"]["v"]
    d = max_abs_diff(a, b)
    print(f"      max-abs-diff (global-sparse vs per-layer-K uniform): {d:.2e}")
    ok &= (d == 0.0)
    print(f"      {'PASS' if d == 0.0 else 'FAIL'} (expect 0.00e+00)")

    # ---- 3: real schedule forward + backward + grad flow ----
    print("[3/4] front_plateau_then_taper 7s7d: forward+backward+grad-flow")
    M = build(1, sparse_attention=True,
              layer_sparse_mask=MASK_7S7D, layer_K_splits=KSPLITS_FPT).to(dtype=dtype)
    M.train()
    assert sum(M.layer_sparse_mask) == len(M.layer_K_splits) == 7
    batch = make_batch(3, 80, "cpu", dtype)
    out = M(batch)
    v = out["bb_ca"]["v"]
    print(f"      output shape {tuple(v.shape)} finite={torch.isfinite(v).all().item()}")
    ok &= (tuple(v.shape) == (3, 80, 3)) and bool(torch.isfinite(v).all())
    loss = (v ** 2).mean()
    loss.backward()
    g_pair = next(p.grad for p in M.pair_repr_builder.parameters() if p.grad is not None)
    g_l0 = any(p.grad is not None and p.grad.abs().sum() > 0
               for p in M.transformer_layers[0].parameters())     # sparse layer
    g_l13 = any(p.grad is not None and p.grad.abs().sum() > 0
                for p in M.transformer_layers[13].parameters())   # dense layer
    print(f"      grad → pair_repr_builder: {g_pair is not None}, "
          f"sparse layer0: {g_l0}, dense layer13: {g_l13}, loss={loss.item():.4f}")
    ok &= (g_pair is not None) and g_l0 and g_l13 and bool(torch.isfinite(loss))

    # ---- 4: short protein (N < max K=56) ----
    print("[4/4] short protein N=24 (< max K=56): padding-slot guard")
    try:
        with torch.no_grad():
            v_short = M.eval()(make_batch(2, 24, "cpu", dtype))["bb_ca"]["v"]
        short_ok = tuple(v_short.shape) == (2, 24, 3) and bool(torch.isfinite(v_short).all())
        print(f"      shape {tuple(v_short.shape)} finite={bool(torch.isfinite(v_short).all())}")
    except Exception as e:  # noqa: BLE001
        short_ok = False
        print(f"      FAIL: {type(e).__name__}: {e}")
    ok &= short_ok

    print("\n" + ("ALL HYBRID SMOKE CHECKS PASSED" if ok else "SMOKE CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
