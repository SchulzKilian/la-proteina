"""Load a TopKRouter / TopKRouterT checkpoint as a frozen module for Move-2
trunk training.

The router ckpt schema (from script_utils/train_router_move1.py and
train_router_move1_inflight.py) is:
    {
        "state_dict": <router.state_dict()>,
        "config": {
            "trunk_dim": int,
            "hidden_dim": int,
            "score_dim": int,
            "n_layers": int,
            "n_heads": int,
            "use_t_emb": bool,
            "pair_features": bool,
            "mlp_block": bool,
            "mlp_dim": int,
        },
        ...
    }

Architecture is inferred from "config" — we don't trust the .pt's class name.
"""
from __future__ import annotations

from typing import Optional

import torch

from proteinfoundation.nn.modules.router import TopKRouter, TopKRouterT


def load_frozen_router(
    ckpt_path: str,
    map_location: Optional[str | torch.device] = "cpu",
) -> torch.nn.Module:
    """Load the router, instantiate the matching architecture, set all params
    requires_grad=False, put in eval() mode. Returns the module.

    The caller is responsible for moving the returned module to the right
    device (e.g. `.to(device)`); doing that here would couple this helper to
    a CUDA-only environment.

    Raises informative errors if architecture fields are missing.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    cfg = ckpt.get("config", None)
    if cfg is None:
        raise ValueError(
            f"Router ckpt {ckpt_path} has no 'config' key — likely a different "
            f"router format. Expected schema: {{'state_dict': ..., 'config': "
            f"{{...}}, ...}}."
        )
    required = ("trunk_dim", "hidden_dim", "score_dim", "n_layers", "n_heads")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(
            f"Router ckpt {ckpt_path} config missing required fields: {missing}. "
            f"Got: {sorted(cfg.keys())}"
        )

    common_kwargs = dict(
        trunk_dim=int(cfg["trunk_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        score_dim=int(cfg["score_dim"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        pair_features=bool(cfg.get("pair_features", False)),
        mlp_block=bool(cfg.get("mlp_block", False)),
        mlp_dim=int(cfg.get("mlp_dim", 256)),
    )
    use_t_emb = bool(cfg.get("use_t_emb", False))
    if use_t_emb:
        # TopKRouterT — pass t_emb_dim if it was saved (older ckpts default to 64).
        common_kwargs["t_emb_dim"] = int(cfg.get("t_emb_dim", 64))
        model = TopKRouterT(**common_kwargs)
    else:
        model = TopKRouter(**common_kwargs)

    missing_keys, unexpected_keys = model.load_state_dict(ckpt["state_dict"], strict=True)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Router state_dict mismatch loading {ckpt_path}:\n"
            f"  missing:    {missing_keys}\n"
            f"  unexpected: {unexpected_keys}"
        )

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    step_loaded = ckpt.get("step", ckpt.get("best_step", "?"))
    best_recall = ckpt.get("recall@64", ckpt.get("best_recall@64", float("nan")))
    print(
        f"[load_frozen_router] {ckpt_path}\n"
        f"  arch: trunk={common_kwargs['trunk_dim']} hidden={common_kwargs['hidden_dim']} "
        f"score={common_kwargs['score_dim']} L={common_kwargs['n_layers']} H={common_kwargs['n_heads']} "
        f"pair={common_kwargs['pair_features']} t_emb={use_t_emb} mlp={common_kwargs['mlp_block']}\n"
        f"  params={n_params:,}  step={step_loaded}  best_recall@64={best_recall}"
    )
    return model
