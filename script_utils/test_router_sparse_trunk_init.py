"""End-to-end smoke test of the router-sparse trunk on CPU.

Builds a LocalLatentsTransformer with router_sparse_K=64, attaches a frozen
router from a checkpoint, and runs a forward pass. Verifies:
  1. attach_router enforces frozen-router invariant.
  2. forward path doesn't error on a tiny CPU batch.
  3. Output shapes match the canonical dense trunk's expectations.
  4. Gradients flow only through trainable trunk params (not the router).
"""
import sys

REPO = "/home/ks2218/la-proteina"
sys.path.insert(0, REPO)

import torch
from omegaconf import OmegaConf

from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer
from script_utils.load_frozen_router import load_frozen_router


def main():
    # Use the canonical nn config + router_sparse_K=64.
    cfg = OmegaConf.load(f"{REPO}/configs/nn/ca_only_router_sparse_K64_160M.yaml")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg.pop("name", None)

    trunk = LocalLatentsTransformer(**cfg)
    n_trunk_params = sum(p.numel() for p in trunk.parameters() if p.requires_grad)
    print(f"[init] trunk params (trainable): {n_trunk_params:,}")

    router = load_frozen_router(
        f"{REPO}/results/router_move1_inflight/router_final.pt",
        map_location="cpu",
    )
    trunk.attach_router(router)
    n_after = sum(p.numel() for p in trunk.parameters() if p.requires_grad)
    assert n_after == n_trunk_params, (
        f"Trainable param count changed after attach_router: {n_trunk_params} -> {n_after}. "
        f"Router params are leaking into the trainable set."
    )
    n_total = sum(p.numel() for p in trunk.parameters())
    print(f"[init] total params after attach_router: {n_total:,} "
          f"(trainable still {n_after:,}; frozen router = {n_total - n_after:,})")

    # Build a small fake input matching what proteina.call_nn produces.
    B, N = 2, 80
    inp = {
        "x_t": {"bb_ca": torch.randn(B, N, 3)},
        "x_sc": {"bb_ca": torch.randn(B, N, 3)},
        "t": {"bb_ca": torch.tensor([0.30, 0.70])},
        "mask": torch.ones(B, N, dtype=torch.bool),
        # Optional features the FeatureFactory consumes — pass zeros / ones to satisfy them.
        "ca_coors_nm": torch.randn(B, N, 3),
        "residue_type": torch.zeros(B, N, dtype=torch.long),
    }

    trunk.eval()
    with torch.no_grad():
        out = trunk(inp)
    print(f"[forward] output keys: {list(out.keys())}")
    assert "bb_ca" in out
    out_v = out["bb_ca"]["v"]
    print(f"[forward] bb_ca.v shape: {tuple(out_v.shape)}; finite={torch.isfinite(out_v).all().item()}")
    assert out_v.shape == (B, N, 3), f"Expected ({B},{N},3), got {tuple(out_v.shape)}"
    assert torch.isfinite(out_v).all(), "non-finite output"

    # Gradient check: backprop through trunk, verify router has zero grad.
    trunk.train()
    out = trunk(inp)
    loss = out["bb_ca"]["v"].pow(2).mean()
    loss.backward()
    router_grad_max = 0.0
    for n, p in trunk._router.named_parameters():
        if p.grad is not None:
            router_grad_max = max(router_grad_max, p.grad.abs().max().item())
    print(f"[grad] max |grad| in router params: {router_grad_max:.3e} "
          f"(must be 0 since router params have requires_grad=False)")
    assert router_grad_max == 0.0
    trunk_grad_max = 0.0
    trunk_grad_count = 0
    for n, p in trunk.named_parameters():
        if p.requires_grad and p.grad is not None:
            trunk_grad_max = max(trunk_grad_max, p.grad.abs().max().item())
            trunk_grad_count += 1
    print(f"[grad] max |grad| in trainable trunk params: {trunk_grad_max:.3e} "
          f"({trunk_grad_count} tensors)")
    assert trunk_grad_max > 0.0
    print("\n[OK] router-sparse trunk init + forward + grad smoke test passed.")


if __name__ == "__main__":
    main()
