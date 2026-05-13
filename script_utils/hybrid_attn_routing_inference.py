"""Inference-time hybrid: §11 sparse + per-residue ATTENTION-derived K-set from dense.

Sibling of `hybrid_grad_routing_inference.py`. Same scaffolding, same sparse
ckpt, same B=2 pad / paired-noise / full_simulation call. The only difference:
the K-set is taken from dense's softmax attention at a SPECIFIC (layer, head)
cell, not from per-query gradient saliency.

Rationale (E062 follow-up): the gradient-routing hybrid was net-negative
(+1.15 Å mean Δ under every_M=50). Two possible explanations:
  (a) gradient saliency is the wrong routing signal (high gradient doesn't
      mean "should be in K-set" — it could be loss-influence flowing through
      residual stream / FFN, not through attention routing);
  (b) §11's trained weights can't generalize to ANY foreign K-set distribution
      without retraining (train/inference mismatch on the K-set itself).

E061 showed max-over-(l, h) Jaccard between gradient top-16 and attention
top-16 was 0.74-0.91 — so they're approximately the same signal at the right
(l, h). If THIS attention-derived hybrid ALSO fails → explanation (b)
(train/inference mismatch); retrain is mandatory. If it succeeds → (a)
(gradient was wrong signal; attention-distilled routing would work).

Default target (layer, head) = L1 H7 — the top winner in E061's cross-metric
audit (18.3 % of queries). Configurable via --target_layer / --target_head.

Mechanism:
  - For each K-set refresh, run ONE dense forward (no backward) and hook
    `PairBiasAttention._attn` to capture the softmax attention pattern at
    the target (layer, head). The top-K attended residues per query become
    the K-set for §11's sparse forward at this step.
  - K-set update strategies are the same as the gradient version
    (frozen_t05 / t_grid_5 / every_M). Cost is dominated by one dense
    forward per refresh — ~0.5-1s at L=200, ~200× cheaper than the
    per-query backward loop in the gradient version. every_M=1 (per step)
    is feasible here in ~1 h wall.

N=3 × L∈{50, 100, 200} × nsteps=400. Compute:
  - frozen_t05: ~5-15 min (one dense forward per protein).
  - t_grid_5:   ~15-30 min.
  - every_M=50: ~30-60 min.
  - every_M=1:  ~1-2 h.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = "/home/ks2218/la-proteina"
sys.path.insert(0, REPO)

import lightning as L
import torch
from dotenv import load_dotenv
from loguru import logger

from proteinfoundation.proteina import Proteina
from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer
from proteinfoundation.nn.modules.pair_bias_attn import PairBiasAttention


DENSE_CKPT_DEFAULT = (
    "/rds/user/ks2218/hpc-work/store/test_ca_only_diffusion/1776805213/"
    "checkpoints/best_val_00000026_000000002646.ckpt"
)
SPARSE_CKPT_DEFAULT = (
    "/rds/user/ks2218/hpc-work/store/ca_only_sparse_K64_curriculum_self/"
    "1778188245/checkpoints/best_val_00000013_000000001385.ckpt"
)

T_GRID_5_DEFAULT = (0.1, 0.3, 0.5, 0.7, 0.9)

# E061 top winner. Configurable via args.
DEFAULT_TARGET_LAYER = 1
DEFAULT_TARGET_HEAD = 7


# ---------------------------------------------------------------------------- #
# Attention K-set: capture top-K attended residues from a specific (layer, head)
# of dense via a forward-time hook on PairBiasAttention._attn.
# ---------------------------------------------------------------------------- #
@torch.no_grad()
def compute_attention_K_set(
    dense_model: Proteina,
    x_t_bb: torch.Tensor,      # [B, n_pad, 3] current x_t for bb_ca
    mask: torch.Tensor,        # [B, n_pad] bool
    t_val: float,
    K: int,
    coords_nm: torch.Tensor,
    coord_mask: torch.Tensor,
    residue_type: torch.Tensor,
    target_layer: int,
    target_head: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """For each real residue i in batch slot 0, take its top-K attended
    residues from dense's softmax attention at (target_layer, target_head).

    Returns:
      neighbor_idx[B_in, n_pad, K] — top-K indices per residue (broadcast over batch)
      slot_valid[B_in, n_pad, K]   — True for real-residue slots, False for padding
    """
    B_in, n_pad, _ = x_t_bb.shape
    device = x_t_bb.device
    real_len = int(mask[0].sum().item())

    # Same row-0-only logic as the gradient version: run_sampling_one duplicates
    # batch slot 0 into slot 1 to satisfy full_simulation's squeeze+assert;
    # we only need to compute K-set for slot 0 and broadcast.
    x_t_bb_row0 = x_t_bb[:1]
    mask_row0 = mask[:1]
    coords_nm_row0 = coords_nm[:1] if coords_nm.shape[0] >= 1 else coords_nm
    coord_mask_row0 = coord_mask[:1] if coord_mask.shape[0] >= 1 else coord_mask
    residue_type_row0 = residue_type[:1] if residue_type.shape[0] >= 1 else residue_type

    # Capture target attention pattern via class-wide monkey patch on _attn.
    # Sparse uses _attn_sparse (different code path), so patching _attn only
    # affects dense's forward — safe to do inside the sampling loop.
    captured: Dict[str, Optional[Tuple[torch.Tensor, int]]] = {"result": None}

    from einops import rearrange
    from torch import einsum

    max_neg_value = lambda x: torch.finfo(x.dtype).min  # noqa: E731
    original_attn = PairBiasAttention._attn

    def hooked_attn(self_attn, q, k, v, b, mask_arg):
        # Exact reproduction of PairBiasAttention._attn (dense path).
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self_attn.scale
        if mask_arg is not None:
            mask_rs = rearrange(mask_arg, "b i j -> b () i j")
            sim = sim.masked_fill(~mask_rs, max_neg_value(sim))
        attn = torch.softmax(sim + b, dim=-1).nan_to_num(0.0)  # [B, H, N, N]
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        layer_idx = getattr(self_attn, "_layer_idx", -1)
        if layer_idx == target_layer:
            # attn shape: [B, H, N, N]; we want head=target_head, batch=0
            attn_lh = attn[0, target_head, :real_len, :real_len].float()  # [N_real, N_real]
            # Mask invalid keys (the dense mask already does this, but defensive
            # safety here).
            mr = mask_row0[0, :real_len].float()
            attn_masked = attn_lh * mr[None, :] + (-1e9) * (1.0 - mr[None, :])
            Keff = min(K, real_len)
            top_idx = attn_masked.topk(k=Keff, dim=-1).indices  # [N_real, Keff]
            captured["result"] = (top_idx.detach(), Keff)
        return out

    PairBiasAttention._attn = hooked_attn
    try:
        # Tag dense's layers so the hook knows where it is.
        for i, layer in enumerate(dense_model.nn.transformer_layers):
            layer.mhba.mha._layer_idx = i

        batch = {
            "coords_nm": coords_nm_row0,
            "coord_mask": coord_mask_row0,
            "residue_type": residue_type_row0,
            "mask": mask_row0,
            "x_t": {"bb_ca": x_t_bb_row0},
            "t": {"bb_ca": torch.full((1,), float(t_val), device=device)},
        }
        _ = dense_model.call_nn(batch, n_recycle=0)
    finally:
        PairBiasAttention._attn = original_attn

    assert captured["result"] is not None, (
        f"target_layer={target_layer} not in dense.transformer_layers "
        f"(have {len(dense_model.nn.transformer_layers)})"
    )
    top_idx_real, Keff = captured["result"]

    # Pad to K if real_len < K.
    if Keff < K:
        pad_idx = torch.zeros(real_len, K - Keff, device=device, dtype=torch.long)
        top_idx_real = torch.cat([top_idx_real, pad_idx], dim=-1)

    neighbor_idx = torch.zeros(1, n_pad, K, device=device, dtype=torch.long)
    neighbor_idx[0, :real_len] = top_idx_real
    slot_valid = torch.zeros(1, n_pad, K, device=device, dtype=torch.bool)
    slot_valid[0, :real_len, :Keff] = True

    if B_in != 1:
        neighbor_idx = neighbor_idx.expand(B_in, n_pad, K).contiguous()
        slot_valid = slot_valid.expand(B_in, n_pad, K).contiguous()

    return neighbor_idx.detach(), slot_valid.detach()


# ---------------------------------------------------------------------------- #
# State + patch (same logic as the gradient version, calls the attention K-set).
# ---------------------------------------------------------------------------- #
class HybridRoutingState:
    def __init__(self, strategy: str, t_grid: Tuple[float, ...], M_steps: int):
        self.strategy = strategy
        self.t_grid = tuple(sorted(t_grid))
        self.M_steps = M_steps
        self.cached_idx: Optional[torch.Tensor] = None
        self.cached_slot_valid: Optional[torch.Tensor] = None
        self.cached_anchor_t: Optional[float] = None
        self.step_counter = 0
        self.frozen = False

    def needs_refresh(self, current_t: float) -> Tuple[bool, Optional[float]]:
        if self.strategy == "frozen_t05":
            return (not self.frozen, 0.5)
        if self.strategy == "t_grid_5":
            nearest = min(self.t_grid, key=lambda tg: abs(tg - current_t))
            if self.cached_anchor_t is None or abs(self.cached_anchor_t - nearest) > 1e-6:
                return (True, nearest)
            return (False, None)
        if self.strategy.startswith("every_"):
            should = (self.step_counter % self.M_steps == 0) or (self.cached_idx is None)
            return (should, current_t)
        raise ValueError(f"Unknown strategy {self.strategy}")


def patch_build_neighbor_idx(
    sparse_nn: LocalLatentsTransformer,
    dense_model: Proteina,
    state: HybridRoutingState,
    K_to_use: int,
    coords_holder: Dict[str, torch.Tensor],
    target_layer: int,
    target_head: int,
):
    orig_build = sparse_nn._build_neighbor_idx

    def hybrid_build(self, x_t, mask, t, **kwargs):
        current_t = float(t.flatten()[0].item()) if isinstance(t, torch.Tensor) else float(t)
        refresh, anchor_t = state.needs_refresh(current_t)
        if refresh:
            new_idx, new_slot_valid = compute_attention_K_set(
                dense_model=dense_model,
                x_t_bb=x_t,
                mask=mask,
                t_val=anchor_t,
                K=K_to_use,
                coords_nm=coords_holder["coords_nm"],
                coord_mask=coords_holder["coord_mask"],
                residue_type=coords_holder["residue_type"],
                target_layer=target_layer,
                target_head=target_head,
            )
            state.cached_idx = new_idx
            state.cached_slot_valid = new_slot_valid
            state.cached_anchor_t = anchor_t
            if state.strategy == "frozen_t05":
                state.frozen = True

        state.step_counter += 1
        return state.cached_idx, state.cached_slot_valid

    sparse_nn._build_neighbor_idx = hybrid_build.__get__(sparse_nn, type(sparse_nn))
    return orig_build


# ---------------------------------------------------------------------------- #
# Sampling — same body as the gradient sibling. predict_for_sampling +
# sampling_model_args, B=2 pad workaround.
# ---------------------------------------------------------------------------- #
def run_sampling_one(
    sparse_model: Proteina,
    coords_nm: torch.Tensor,
    coord_mask: torch.Tensor,
    residue_type: torch.Tensor,
    mask: torch.Tensor,
    nsteps: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    from functools import partial as _partial
    from omegaconf import OmegaConf as _OC
    torch.manual_seed(seed)
    B, n_pad = mask.shape
    assert B == 1, f"run_sampling_one is per-sample (B=1); got B={B}"

    B_use = 2
    mask_use = mask.expand(B_use, n_pad).contiguous()

    sampling_model_args = _OC.create({
        "bb_ca": {
            "schedule": {"mode": "log", "p": 2.0},
            "gt": {"mode": "1/t", "p": 1.0, "clamp_val": None},
            "simulation_step_params": {
                "sampling_mode": "sc",
                "sc_scale_noise": 0.1,
                "sc_scale_score": 1.0,
                "t_lim_ode": 0.98,
                "t_lim_ode_below": 0.02,
                "center_every_step": True,
            },
        },
    })
    batch = {
        "nsamples": B_use,
        "nres": n_pad,
        "mask": mask_use,
    }
    fn_predict_for_sampling = _partial(
        sparse_model.predict_for_sampling, n_recycle=0
    )
    sc_neighbors_active = sparse_model.cfg_exp.training.get("sc_neighbors", False)
    with torch.no_grad():
        gen_samples, _extra_info = sparse_model.fm.full_simulation(
            batch=batch,
            predict_for_sampling=fn_predict_for_sampling,
            nsteps=nsteps,
            nsamples=B_use,
            n=n_pad,
            self_cond=True,
            sampling_model_args=sampling_model_args,
            device=device,
            save_trajectory_every=0,
            guidance_w=1.0,
            ag_ratio=0.0,
            steering_guide=None,
            sc_neighbors_active=sc_neighbors_active,
            sc_neighbors_bootstrap=True,
        )

    if isinstance(gen_samples, dict):
        if "bb_ca" in gen_samples:
            final = gen_samples["bb_ca"]
        else:
            final = next(iter(gen_samples.values()))
            if isinstance(final, dict) and "bb_ca" in final:
                final = final["bb_ca"]
    else:
        final = gen_samples

    return final[0]


def save_pdb_ca_only(coords_ca: torch.Tensor, mask: torch.Tensor, residue_type: torch.Tensor, path: str):
    AA_THREE = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK",
    ]
    coords_ang = coords_ca.detach().cpu().numpy() * 10.0
    mask_np = mask.detach().cpu().numpy().astype(bool)
    rtype_np = residue_type.detach().cpu().numpy().astype(int)
    lines = []
    atom_serial = 1
    chain = "A"
    for i in range(len(mask_np)):
        if not mask_np[i]:
            continue
        rt = rtype_np[i]
        aa3 = AA_THREE[rt] if 0 <= rt < len(AA_THREE) else "UNK"
        x, y, z = coords_ang[i]
        lines.append(
            f"ATOM  {atom_serial:>5d}  CA  {aa3} {chain}{i+1:>4d}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C"
        )
        atom_serial += 1
    lines.append("END")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------- #
# Main.
# ---------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--dense_ckpt", default=DENSE_CKPT_DEFAULT)
    parser.add_argument("--sparse_ckpt", default=SPARSE_CKPT_DEFAULT)
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--lengths", type=str, default="50,100,200")
    parser.add_argument("--K", type=int, default=64,
                        help="K-set size to inject; default 64 matches §11.")
    parser.add_argument(
        "--target_layer", type=int, default=DEFAULT_TARGET_LAYER,
        help=f"Dense layer index to read attention from. Default {DEFAULT_TARGET_LAYER} "
             f"(L{DEFAULT_TARGET_LAYER} H{DEFAULT_TARGET_HEAD} was E061's top winner — "
             "18%% of queries' best-match cell).",
    )
    parser.add_argument(
        "--target_head", type=int, default=DEFAULT_TARGET_HEAD,
        help="Dense head index within --target_layer.",
    )
    parser.add_argument(
        "--routing_strategy",
        choices=["frozen_t05", "t_grid_5", "every_M"],
        default="every_M",
        help="K-set update strategy. attention is ~200x cheaper than gradient, "
             "so every_M=50 (or M=1) is realistic here.",
    )
    parser.add_argument("--M_steps", type=int, default=50,
                        help="For every_M strategy: refresh K-set every M ODE steps.")
    parser.add_argument("--nsteps", type=int, default=400,
                        help="ODE steps. CLAUDE.md hard rule.")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--baseline_only", action="store_true")
    parser.add_argument("--hybrid_only", action="store_true")
    parser.add_argument("--out_dir", default="results/hybrid_attn_routing")
    args = parser.parse_args()

    load_dotenv()
    L.seed_everything(args.seed)
    assert args.nsteps == 400, "nsteps must be 400 for designability per CLAUDE.md hard rule."

    lengths = [int(x) for x in args.lengths.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Inference requires CUDA."

    out_root = Path(args.out_dir) / args.label
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 80)
    logger.info(f"Hybrid attention-routing inference | label={args.label}")
    logger.info(f"  dense_ckpt:    {args.dense_ckpt}")
    logger.info(f"  sparse_ckpt:   {args.sparse_ckpt}")
    logger.info(f"  N={args.n_samples} × L={lengths} × nsteps={args.nsteps}")
    logger.info(f"  K={args.K}, target=(layer={args.target_layer}, head={args.target_head})")
    logger.info(f"  routing_strategy={args.routing_strategy} (M_steps={args.M_steps})")
    logger.info(f"  arms: baseline={not args.hybrid_only}, hybrid={not args.baseline_only}")
    logger.info("=" * 80)

    logger.info("Loading dense model...")
    dense_model = Proteina.load_from_checkpoint(
        args.dense_ckpt, strict=False, autoencoder_ckpt_path=None
    )
    dense_model.to(device).eval()
    for p in dense_model.parameters():
        p.requires_grad_(False)

    logger.info("Loading sparse model (§11)...")
    sparse_model = Proteina.load_from_checkpoint(
        args.sparse_ckpt, strict=False, autoencoder_ckpt_path=None
    )
    sparse_model.to(device).eval()
    for p in sparse_model.parameters():
        p.requires_grad_(False)

    # Unwrap torch.compile (same workaround as the gradient sibling).
    if hasattr(sparse_model.nn, "_orig_mod"):
        logger.info("  unwrapping sparse_model.nn from torch.compile.")
        sparse_model.nn = sparse_model.nn._orig_mod
    if hasattr(dense_model.nn, "_orig_mod"):
        logger.info("  unwrapping dense_model.nn from torch.compile.")
        dense_model.nn = dense_model.nn._orig_mod

    sp_cfg = sparse_model.cfg_exp.get("nn", {})
    assert sp_cfg.get("sparse_attention", False), "Wrong sparse ckpt — sparse_attention=False"
    de_cfg = dense_model.cfg_exp.get("nn", {})
    assert not de_cfg.get("sparse_attention", False), "Wrong dense ckpt — sparse_attention=True"

    n_layers_dense = len(dense_model.nn.transformer_layers)
    n_heads_dense = dense_model.cfg_exp.get("nn", {}).get("nheads", 12)
    assert 0 <= args.target_layer < n_layers_dense, (
        f"target_layer={args.target_layer} out of range [0, {n_layers_dense})"
    )
    assert 0 <= args.target_head < n_heads_dense, (
        f"target_head={args.target_head} out of range [0, {n_heads_dense})"
    )

    state = HybridRoutingState(
        strategy=args.routing_strategy,
        t_grid=T_GRID_5_DEFAULT,
        M_steps=args.M_steps,
    )
    coords_holder: Dict[str, torch.Tensor] = {}

    original_build = sparse_model.nn._build_neighbor_idx

    def install_hybrid_patch():
        state.cached_idx = None
        state.cached_slot_valid = None
        state.cached_anchor_t = None
        state.step_counter = 0
        state.frozen = False
        patch_build_neighbor_idx(
            sparse_nn=sparse_model.nn,
            dense_model=dense_model,
            state=state,
            K_to_use=args.K,
            coords_holder=coords_holder,
            target_layer=args.target_layer,
            target_head=args.target_head,
        )

    def uninstall_hybrid_patch():
        sparse_model.nn._build_neighbor_idx = original_build

    metadata: List[dict] = []
    for L_target in lengths:
        n_pad = max(L_target + 16, 64)
        for prot_idx in range(args.n_samples):
            seed = args.seed + prot_idx * 1000 + L_target
            logger.info(f"--- L={L_target} sample {prot_idx} (seed={seed}) ---")

            coords_nm = torch.zeros(1, n_pad, 37, 3, device=device)
            coord_mask = torch.zeros(1, n_pad, 37, dtype=torch.bool, device=device)
            coord_mask[0, :L_target, 1] = True
            residue_type = torch.zeros(1, n_pad, dtype=torch.long, device=device)
            mask = torch.zeros(1, n_pad, dtype=torch.bool, device=device)
            mask[0, :L_target] = True

            coords_holder["coords_nm"] = coords_nm
            coords_holder["coord_mask"] = coord_mask
            coords_holder["residue_type"] = residue_type

            if not args.hybrid_only:
                uninstall_hybrid_patch()
                t0 = time.time()
                final_baseline = run_sampling_one(
                    sparse_model=sparse_model,
                    coords_nm=coords_nm,
                    coord_mask=coord_mask,
                    residue_type=residue_type,
                    mask=mask,
                    nsteps=args.nsteps,
                    seed=seed,
                    device=device,
                )
                t_b = time.time() - t0
                bdir = out_root / "baseline" / f"L{L_target}"
                bdir.mkdir(parents=True, exist_ok=True)
                bpath = bdir / f"sample_{prot_idx}.pdb"
                save_pdb_ca_only(
                    final_baseline[:L_target], mask[0, :L_target],
                    residue_type[0, :L_target], str(bpath)
                )
                logger.info(f"  baseline saved {bpath}  ({t_b:.1f}s)")
                metadata.append({
                    "arm": "baseline",
                    "L": L_target,
                    "sample_idx": prot_idx,
                    "seed": seed,
                    "path": str(bpath),
                    "wall_s": t_b,
                })

            if not args.baseline_only:
                install_hybrid_patch()
                t0 = time.time()
                final_hybrid = run_sampling_one(
                    sparse_model=sparse_model,
                    coords_nm=coords_nm,
                    coord_mask=coord_mask,
                    residue_type=residue_type,
                    mask=mask,
                    nsteps=args.nsteps,
                    seed=seed,
                    device=device,
                )
                t_h = time.time() - t0
                hdir = out_root / "hybrid" / f"L{L_target}"
                hdir.mkdir(parents=True, exist_ok=True)
                hpath = hdir / f"sample_{prot_idx}.pdb"
                save_pdb_ca_only(
                    final_hybrid[:L_target], mask[0, :L_target],
                    residue_type[0, :L_target], str(hpath)
                )
                logger.info(f"  hybrid   saved {hpath}  ({t_h:.1f}s, "
                            f"target=(L{args.target_layer} H{args.target_head}))")
                metadata.append({
                    "arm": "hybrid",
                    "L": L_target,
                    "sample_idx": prot_idx,
                    "seed": seed,
                    "routing_strategy": args.routing_strategy,
                    "K": args.K,
                    "target_layer": args.target_layer,
                    "target_head": args.target_head,
                    "path": str(hpath),
                    "wall_s": t_h,
                })
                uninstall_hybrid_patch()

    meta_path = out_root / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "label": args.label,
                "dense_ckpt": args.dense_ckpt,
                "sparse_ckpt": args.sparse_ckpt,
                "K": args.K,
                "nsteps": args.nsteps,
                "n_samples": args.n_samples,
                "lengths": lengths,
                "routing_strategy": args.routing_strategy,
                "M_steps": args.M_steps,
                "target_layer": args.target_layer,
                "target_head": args.target_head,
                "seed": args.seed,
                "samples": metadata,
            },
            f,
            indent=2,
        )
    logger.info(f"Metadata written to {meta_path}")
    logger.info("Done. Next: run eval_hybrid_grad_routing.py (or evaluate.py) on the saved PDBs.")


if __name__ == "__main__":
    main()
