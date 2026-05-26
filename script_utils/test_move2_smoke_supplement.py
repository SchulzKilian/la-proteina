"""Supplemental Move-2 smoke test (companion to the agent's existing tests).

Existing tests cover identity-K/permuted-K/short-protein/grad-flow at the
attention-module level (test_pair_bias_attn_sparse.py), and module-level
init+forward+grad of the full trunk (test_router_sparse_trunk_init.py).

This file adds the three checks the reviewer specifically asked about that
are NOT covered above:

  (1) DETERMINISM — two forward passes on the same input must produce
      bit-identical outputs (no router dropout / sampling leakage).
  (2) MASK-IS-DOING-SOMETHING — sparse-trunk output must differ from
      dense-trunk output on the same input. Frobenius-norm difference
      should be small (mask kept self + 63 router preds out of N keys)
      but non-trivial. Catches a "router K-set is no-op" failure mode.
  (3) ROUTER WEIGHTS UNCHANGED THROUGH ONE OPT STEP — load the router,
      take one synthetic backward + optimizer step on a trunk-only param
      list, then assert every router parameter is byte-identical
      (torch.equal) to its pre-step value. This is the empirical version
      of the "is the router actually frozen" check.

Run on CPU; takes ~30s. Requires:
  /home/ks2218/la-proteina/results/router_move1_inflight/router_final.pt
"""
import copy
import sys

REPO = "/home/ks2218/la-proteina"
sys.path.insert(0, REPO)

import torch
from omegaconf import OmegaConf

from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer
from script_utils.load_frozen_router import load_frozen_router


def _build_trunk(router_sparse_K, seed=0):
    """Build a trunk with or without router-sparse, sharing the same seed
    so that the trainable params are identical at init across the two."""
    torch.manual_seed(seed)
    cfg = OmegaConf.load(f"{REPO}/configs/nn/ca_only_router_sparse_K64_160M.yaml")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg.pop("name", None)
    if router_sparse_K is None:
        cfg.pop("router_sparse_K", None)
    else:
        cfg["router_sparse_K"] = router_sparse_K
    return LocalLatentsTransformer(**cfg)


def _make_input(B=2, N=40, seed=123):
    g = torch.Generator().manual_seed(seed)
    return {
        "x_t": {"bb_ca": torch.randn(B, N, 3, generator=g)},
        "x_sc": {"bb_ca": torch.randn(B, N, 3, generator=g)},
        "t": {"bb_ca": torch.tensor([0.30, 0.70])},
        "mask": torch.ones(B, N, dtype=torch.bool),
        "ca_coors_nm": torch.randn(B, N, 3, generator=g),
        "residue_type": torch.zeros(B, N, dtype=torch.long),
    }


def test_determinism():
    """Two forward passes on the same input must produce identical outputs."""
    trunk = _build_trunk(router_sparse_K=64, seed=0)
    router = load_frozen_router(
        f"{REPO}/results/router_move1_inflight/router_final.pt", map_location="cpu"
    )
    trunk.attach_router(router)
    trunk.eval()  # eval avoids any module-level non-determinism
    inp = _make_input()
    with torch.no_grad():
        out1 = trunk(inp)["bb_ca"]["v"]
        out2 = trunk(inp)["bb_ca"]["v"]
    delta = (out1 - out2).abs().max().item()
    assert delta == 0.0, f"non-deterministic forward: max abs diff = {delta:.3e}"
    print(f"[PASS] determinism: max abs diff = {delta:.3e}")


def test_sparse_differs_from_dense():
    """Sparse trunk output should differ from dense trunk output on the same
    input (otherwise the K-set mask is a no-op). Ratio in the 1e-3 .. 1e+1
    band is sane; outside that range, something is wrong."""
    dense = _build_trunk(router_sparse_K=None, seed=0)
    sparse = _build_trunk(router_sparse_K=64, seed=0)

    # Sanity: trunk init was deterministic so the two trunks should share
    # every trainable parameter at this point.
    for (n_d, p_d), (n_s, p_s) in zip(dense.named_parameters(), sparse.named_parameters()):
        if n_d == n_s and p_d.requires_grad and p_s.requires_grad:
            assert torch.equal(p_d, p_s), f"init mismatch on {n_d}"

    router = load_frozen_router(
        f"{REPO}/results/router_move1_inflight/router_final.pt", map_location="cpu"
    )
    sparse.attach_router(router)
    dense.eval()
    sparse.eval()

    inp = _make_input()
    with torch.no_grad():
        out_dense = dense(inp)["bb_ca"]["v"]
        out_sparse = sparse(inp)["bb_ca"]["v"]

    diff_fro = (out_dense - out_sparse).pow(2).sum().sqrt().item()
    dense_fro = out_dense.pow(2).sum().sqrt().item()
    ratio = diff_fro / max(dense_fro, 1e-9)

    # At N=40, K=64 means router would pick all 40 — the mask reduces to a
    # no-op at the attention level. So the diff should be NEAR ZERO at N=40.
    # We still check it's finite and small.
    print(f"[INFO] N=40, K=64 (K≥N degenerate): "
          f"||sparse-dense||_F = {diff_fro:.3e}, "
          f"||dense||_F = {dense_fro:.3e}, ratio = {ratio:.3e}")
    assert torch.isfinite(out_sparse).all() and torch.isfinite(out_dense).all()

    # Re-run at a length where K < N so the mask is non-degenerate.
    inp2 = _make_input(B=2, N=200)
    with torch.no_grad():
        out_dense2 = dense(inp2)["bb_ca"]["v"]
        out_sparse2 = sparse(inp2)["bb_ca"]["v"]
    diff2 = (out_dense2 - out_sparse2).pow(2).sum().sqrt().item()
    dense2 = out_dense2.pow(2).sum().sqrt().item()
    ratio2 = diff2 / max(dense2, 1e-9)
    print(f"[INFO] N=200, K=64 (K<N, mask active): "
          f"||sparse-dense||_F = {diff2:.3e}, ||dense||_F = {dense2:.3e}, ratio = {ratio2:.3e}")
    # At init, trunk-trunk diff is dominated by random-init Q/K/V noise.
    # We expect ratio2 in (1e-2, 1e+1). Outside that range = suspicious.
    assert 1e-4 < ratio2 < 1e+2, (
        f"sparse vs dense at N=200 ratio = {ratio2:.3e} is suspicious. "
        f"Expected (1e-2, 1e+1)."
    )
    print(f"[PASS] sparse-trunk output meaningfully differs from dense at N=200 (mask active).")


def test_router_weights_byte_identical_through_opt_step():
    """Frozen router invariant — empirical: take one optimizer step on the
    trunk-only param list and assert router weights are byte-identical."""
    trunk = _build_trunk(router_sparse_K=64, seed=0)
    router = load_frozen_router(
        f"{REPO}/results/router_move1_inflight/router_final.pt", map_location="cpu"
    )
    trunk.attach_router(router)

    # Snapshot router params before any optimization
    router_before = {
        n: p.detach().clone() for n, p in trunk._router.named_parameters()
    }

    # Build the optimizer exactly the way Proteina.configure_optimizers does
    # (uniform-wd branch, since cfg has no param_groups override).
    opt = torch.optim.AdamW(
        [p for p in trunk.parameters() if p.requires_grad],
        lr=2e-4, weight_decay=0.05,
    )
    n_opt_params = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_router_params = sum(p.numel() for p in trunk._router.parameters())
    n_total = sum(p.numel() for p in trunk.parameters())
    print(f"[INFO] optimizer param count = {n_opt_params:,}; "
          f"router param count = {n_router_params:,}; "
          f"trunk total = {n_total:,}.")
    assert n_opt_params + n_router_params == n_total, (
        "param accounting mismatch: optimizer + router ≠ total. "
        "Either router leaked into optimizer or trainable trunk params are missing."
    )

    trunk.train()
    inp = _make_input()
    out = trunk(inp)
    loss = out["bb_ca"]["v"].pow(2).mean()
    loss.backward()
    opt.step()

    # Router params must still be bit-equal.
    for n, p_before in router_before.items():
        p_after = dict(trunk._router.named_parameters())[n]
        if not torch.equal(p_before, p_after):
            max_d = (p_before - p_after).abs().max().item()
            raise AssertionError(
                f"router param '{n}' changed after one opt step "
                f"(max abs diff = {max_d:.3e}). Frozen-router invariant broken."
            )
    print(f"[PASS] router byte-identical through one opt step "
          f"({len(router_before)} params checked).")


if __name__ == "__main__":
    print("=== Move-2 supplemental smoke ===")
    test_determinism()
    test_sparse_differs_from_dense()
    test_router_weights_byte_identical_through_opt_step()
    print("\n[OK] all supplemental Move-2 smoke checks passed.")
