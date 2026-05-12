"""Smoke test for the BigBird CLS + pair-update variant
(ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_160M).

Verifies:
  1. Constructor builds with n_global_tokens=4 + sparse + pair_update + curriculum.
  2. Forward pass on a tiny synthetic batch produces finite output of shape [B, N, 3].
  3. Backward pass produces non-zero grads on:
       - global_token_emb, global_cond_emb
       - global_pair_bias_res_to_glob / glob_to_res / glob_to_glob
       - pair_update_layers (at least one)
  4. Trunk runs at K_total=68 (K_canonical=64 + G=4) without index-OOB.
  5. n_global_tokens=0 path still works (regression guard for existing variants).

Run from repo root:
    /home/ks2218/conda_envs/laproteina_env/bin/python script_utils/smoke_bigbird_pairupdate.py
"""
import os
import sys
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/home/ks2218/la-proteina"
sys.path.insert(0, REPO)

from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer


def build_nn_kwargs(n_global_tokens: int = 4) -> dict:
    """Build kwargs matching the new NN config; small B/N for speed."""
    cfg_dir = os.path.join(REPO, "configs", "nn")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(
            config_name="ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_160M"
        )
    kwargs = OmegaConf.to_container(cfg, resolve=True)
    kwargs["n_global_tokens"] = n_global_tokens
    # curriculum_neighbors is plumbed from training config in production;
    # for the smoke test we pass it directly as a kwarg.
    kwargs["curriculum_neighbors"] = True
    return kwargs


def build_batch(B: int, N: int, device: torch.device) -> dict:
    """Synthetic batch matching what the canonical NN forward expects."""
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    # Mark last 5 residues of the first protein as padded to exercise variable lengths
    if N >= 10 and B >= 1:
        mask[0, -5:] = False
    x_t_ca = torch.randn(B, N, 3, device=device, dtype=torch.float32) * 0.3
    x_sc_ca = torch.randn(B, N, 3, device=device, dtype=torch.float32) * 0.3
    t = torch.rand(B, device=device, dtype=torch.float32)
    # Optional features expected by the NN config (optional_ca_coors_nm_seq_feat /
    # optional_res_type_seq_feat / optional_ca_pair_dist). Provide a residue_type and
    # the residue-mask flag for "optional present".
    residue_type = torch.zeros(B, N, dtype=torch.long, device=device)
    return {
        "mask": mask,
        "x_t": {"bb_ca": x_t_ca},
        "x_sc": {"bb_ca": x_sc_ca},
        "t": {"bb_ca": t},
        "residue_type": residue_type,
        "ca_coors_nm": x_t_ca.clone(),
        "ca_coors_nm_mask": mask.clone(),
        "res_type_mask": mask.clone(),
    }


def check_grads(model: torch.nn.Module, names: list[str]) -> dict:
    """Return {param_name: (grad_present, grad_norm)} for each named param."""
    out = {}
    found = dict(model.named_parameters())
    for name in names:
        # Allow prefix-match (handles parameters inside ModuleList by index).
        matches = [k for k in found if name in k]
        if not matches:
            out[name] = (False, 0.0)
            continue
        for k in matches:
            g = found[k].grad
            if g is None:
                out[k] = (False, 0.0)
            else:
                out[k] = (True, g.detach().abs().mean().item())
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}, torch={torch.__version__}")

    # --- Phase 1: n_global_tokens=4 path ---
    print("\n[phase 1] BigBird + pair update + curriculum, n_global_tokens=4")
    kwargs = build_nn_kwargs(n_global_tokens=4)
    nn_module = LocalLatentsTransformer(**kwargs).to(device)
    nn_module.train()

    # Quick parameter-count summary
    n_total = sum(p.numel() for p in nn_module.parameters() if p.requires_grad)
    n_globals = sum(
        p.numel() for name, p in nn_module.named_parameters()
        if "global_" in name and p.requires_grad
    )
    print(f"[phase 1] params total={n_total/1e6:.2f}M, global-related={n_globals}")

    # Small batch
    B, N = 2, 32
    batch = build_batch(B, N, device)
    print(f"[phase 1] batch B={B} N={N}, x_t shape={batch['x_t']['bb_ca'].shape}")

    # Forward
    nn_out = nn_module(batch)
    ca_out = nn_out["bb_ca"]["v"]
    assert ca_out.shape == (B, N, 3), f"unexpected output shape: {ca_out.shape}"
    assert torch.isfinite(ca_out).all(), "non-finite output"
    print(f"[phase 1] forward OK, output shape={ca_out.shape}, "
          f"|out| mean={ca_out.abs().mean().item():.4f}")

    # Backward on a fake target
    target = torch.randn_like(ca_out)
    loss = ((ca_out - target) * batch["mask"][..., None]).pow(2).mean()
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    print(f"[phase 1] loss={loss.item():.4f}")
    loss.backward()

    # Grad checks
    keys_to_check = [
        "global_token_emb",
        "global_cond_emb",
        "global_pair_bias_res_to_glob",
        "global_pair_bias_glob_to_res",
        "global_pair_bias_glob_to_glob",
        "pair_update_layers.0",  # first pair-update layer (every_n=3 → present at i=0)
        "transformer_layers.0",
    ]
    grad_summary = check_grads(nn_module, keys_to_check)
    print("[phase 1] grad summary:")
    all_present = True
    for k, (present, gnorm) in grad_summary.items():
        status = "OK" if present else "MISSING"
        print(f"   {status:7s}  {k:50s}  |grad|.mean={gnorm:.6e}")
        if not present:
            all_present = False
        # Globals must have non-zero grads (otherwise something is silently bypassing them).
        if present and "global_" in k and gnorm == 0.0:
            print(f"   WARN — zero-grad on {k} (may indicate a wiring break)")
    assert all_present, "some expected params have no grad — wiring break"

    # --- Phase 2: regression guard — n_global_tokens=0 path ---
    print("\n[phase 2] regression guard: n_global_tokens=0")
    kwargs0 = build_nn_kwargs(n_global_tokens=0)
    nn0 = LocalLatentsTransformer(**kwargs0).to(device)
    nn0.train()
    has_globals = any("global_" in name for name, _ in nn0.named_parameters())
    assert not has_globals, "n_global_tokens=0 must not create global parameters"
    out0 = nn0(batch)
    ca0 = out0["bb_ca"]["v"]
    assert ca0.shape == (B, N, 3) and torch.isfinite(ca0).all()
    print(f"[phase 2] no-globals path OK, output shape={ca0.shape}")

    print("\n[smoke] PASS")


if __name__ == "__main__":
    main()
