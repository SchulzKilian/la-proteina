"""Inference-time hybrid: §11 sparse + per-residue gradient-derived K-set from dense.

E061 (per-query gradient audit) showed:
  - per-residue (= per query in attention nomenclature) gradient saliency is
    concentrated: mass_top_16 = 0.71 averaged across (proteins, t-values).
  - query-pair Jaccard collapses with L: 0.06 at L=200. Different residues
    genuinely have disjoint important sets at long L — shared K-set is wrong.
  - cross-metric max-over-(l, h) Jaccard with dense attention is 0.74-0.91:
    dense's attention at THE RIGHT (l, h) IS the gradient-derived K-set.

This script tests the hypothesis directly: if we inject the gradient-derived
per-residue K-set into §11 sparse at inference time, can §11 recover
dense-quality samples? If yes, the routing prior is real and the next move
is a learned router that distills it. If no, the K-set isn't the bottleneck.

Mechanism:
  - Load §11 sparse ckpt (rmuumq8v, val 4.19 at step 1385) and canonical
    dense ckpt (best_val_2646).
  - For each (length, sample_idx): run §11's sampling at nsteps=400 (HARD RULE)
    with `_build_neighbor_idx` monkey-patched to use a gradient-derived K-set
    instead of the seq+spatial+random curriculum.
  - K-set update strategy (configurable, default: 'frozen_t05'):
      - frozen_t05: compute K-set ONCE at t=0.5 from a synthetic x_t (noise +
        midway interpolant); freeze for the entire trajectory. Cheapest.
      - t_grid_5: compute K-set at t∈{0.1, 0.3, 0.5, 0.7, 0.9}; at each
        sampling step, use the nearest-t cached K-set. 5 saliency computes
        per protein.
      - every_M (M configurable): recompute K-set every M ODE steps. Most
        expensive; fine-grained adaptation.
  - Also runs a BASELINE: vanilla §11 sparse (default K-set from curriculum)
    on the same proteins + same seeds, for paired A/B.

Per-residue saliency = ‖∂‖v_pred[i]‖₂ / ∂x_t[j]‖₂ for each j, computed via
vector-Jacobian product on the dense model. No ground truth needed.

Output: PDB files in results/hybrid_grad_routing/<label>/ for both arms.
Run scRMSD eval afterwards via evaluate.py.

N=3 × L∈{50, 100, 200} × nsteps=400. Compute:
  - frozen_t05 default: ~30-60 min on 1× A100 (mostly sampling, not saliency).
  - t_grid_5: ~1-2h.
  - every_M=50: ~3-4h.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = "/home/ks2218/la-proteina"
sys.path.insert(0, REPO)

import lightning as L
import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger

from proteinfoundation.proteina import Proteina
from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer


DENSE_CKPT_DEFAULT = (
    "/rds/user/ks2218/hpc-work/store/test_ca_only_diffusion/1776805213/"
    "checkpoints/best_val_00000026_000000002646.ckpt"
)
SPARSE_CKPT_DEFAULT = (
    "/rds/user/ks2218/hpc-work/store/ca_only_sparse_K64_curriculum_self/"
    "1778188245/checkpoints/best_val_00000013_000000001385.ckpt"
)

# Default K-update t-grid for t_grid_5 strategy.
T_GRID_5_DEFAULT = (0.1, 0.3, 0.5, 0.7, 0.9)


# ---------------------------------------------------------------------------- #
# Saliency: compute per-residue gradient-derived K-set via dense.
# ---------------------------------------------------------------------------- #
@torch.enable_grad()
def compute_gradient_K_set(
    dense_model: Proteina,
    x_t_bb: torch.Tensor,      # [B, n_pad, 3] current x_t for bb_ca
    mask: torch.Tensor,        # [B, n_pad] bool
    t_val: float,
    K: int,
    coords_nm: torch.Tensor,   # [B, n_pad, 37, 3] for nn input
    coord_mask: torch.Tensor,  # [B, n_pad, 37]
    residue_type: torch.Tensor,  # [B, n_pad]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """For each real residue i in batch slot 0, compute its top-K residues j
    by ‖∂‖v_pred[i]‖₂ / ∂x_t[j]‖₂ from dense. Returns:
      neighbor_idx[B, n_pad, K] — top-K indices per residue (padding queries
                                  get a placeholder filled with zeros)
      slot_valid[B, n_pad, K]  — True if the index points to a real residue
    """
    B_in, n_pad, _ = x_t_bb.shape
    device = x_t_bb.device
    # run_sampling_one pads B from 1 to 2 by duplication to satisfy
    # full_simulation's squeeze+assert; here we recover row 0 (the real
    # sample) and expand outputs back to B_in.
    real_len = int(mask[0].sum().item())
    B = 1
    x_t_bb_row0 = x_t_bb[:1]      # [1, n_pad, 3]
    mask_row0 = mask[:1]          # [1, n_pad]
    # coords_nm / coord_mask / residue_type are constructed at [1, ...] in
    # the caller — pass as-is. (Dense doesn't care about rows beyond what
    # x_t / mask define.)
    coords_nm_row0 = coords_nm[:1] if coords_nm.shape[0] >= 1 else coords_nm
    coord_mask_row0 = coord_mask[:1] if coord_mask.shape[0] >= 1 else coord_mask
    residue_type_row0 = residue_type[:1] if residue_type.shape[0] >= 1 else residue_type

    # Make x_t a leaf with grad enabled.
    x_t_bb_g = x_t_bb_row0.detach().clone().requires_grad_(True)

    # Build a minimal batch dict for dense.call_nn.
    batch = {
        "coords_nm": coords_nm_row0,
        "coord_mask": coord_mask_row0,
        "residue_type": residue_type_row0,
        "mask": mask_row0,
        "x_t": {"bb_ca": x_t_bb_g},
        "t": {"bb_ca": torch.full((B,), float(t_val), device=device)},
    }
    # Some scaffolding (x_0, x_1) is needed by compute_loss but NOT call_nn.
    # call_nn just runs the network. We need v_pred from nn_out["bb_ca"]["v"].
    nn_out = dense_model.call_nn(batch, n_recycle=0)
    v_pred = nn_out["bb_ca"]["v"]  # [B, n_pad, 3]
    v_pred_real = v_pred[0, :real_len]  # [N_real, 3]

    # Per-query VJP: for each real query i, backprop ‖v_pred[i]‖₂.
    saliencies = torch.zeros(real_len, n_pad, device=device)
    for query_i in range(real_len):
        if x_t_bb_g.grad is not None:
            x_t_bb_g.grad.zero_()
        scalar_i = v_pred_real[query_i].norm()
        retain = (query_i != real_len - 1)
        scalar_i.backward(retain_graph=retain)
        sal_i = x_t_bb_g.grad[0].norm(dim=-1)  # [n_pad]
        saliencies[query_i] = sal_i

    # Mask out padding positions BEFORE top-K (padding shouldn't be a neighbor).
    real_mask_keys = mask[0].float()  # [n_pad]
    saliencies = saliencies * real_mask_keys[None, :] - 1e9 * (1.0 - real_mask_keys[None, :])

    # Top-K per real query.
    Keff = min(K, real_len)
    top_idx_real = saliencies.topk(k=Keff, dim=-1).indices  # [N_real, Keff]
    # If Keff < K (short proteins), pad with zeros (sparse model treats them as
    # padding via slot_valid mask).
    if Keff < K:
        pad_idx = torch.zeros(real_len, K - Keff, device=device, dtype=torch.long)
        top_idx_real = torch.cat([top_idx_real, pad_idx], dim=-1)  # [N_real, K]

    # Build full [B, n_pad, K] neighbor_idx — pad queries get zeros (won't be used).
    neighbor_idx = torch.zeros(B, n_pad, K, device=device, dtype=torch.long)
    neighbor_idx[0, :real_len] = top_idx_real

    # slot_valid[i, k] = True if neighbor_idx[i, k] points at a real residue.
    slot_valid = torch.zeros(B, n_pad, K, device=device, dtype=torch.bool)
    slot_valid[0, :real_len, :Keff] = True  # the real top-K entries
    # Padding columns (Keff..K) are False; padding queries' rows are all False.

    # Broadcast row 0 to all B_in rows (rows 1..B_in-1 are duplicates of row 0
    # under run_sampling_one's pad).
    if B_in != B:
        neighbor_idx = neighbor_idx.expand(B_in, n_pad, K).contiguous()
        slot_valid = slot_valid.expand(B_in, n_pad, K).contiguous()

    return neighbor_idx.detach(), slot_valid.detach()


# ---------------------------------------------------------------------------- #
# Sparse model patching: replace _build_neighbor_idx with a hybrid version
# that consults a cached gradient-derived K-set.
# ---------------------------------------------------------------------------- #
class HybridRoutingState:
    """Mutable state holding the current K-set + bookkeeping for the patched
    _build_neighbor_idx. Attached to the sparse model's nn so the closure
    can read it during sampling."""

    def __init__(self, strategy: str, t_grid: Tuple[float, ...], M_steps: int):
        self.strategy = strategy
        self.t_grid = tuple(sorted(t_grid))
        self.M_steps = M_steps
        self.cached_idx: Optional[torch.Tensor] = None
        self.cached_slot_valid: Optional[torch.Tensor] = None
        self.cached_anchor_t: Optional[float] = None
        self.step_counter = 0
        # For 'frozen_t05', the K-set is precomputed once outside the sampling loop.
        self.frozen = False

    def needs_refresh(self, current_t: float) -> Tuple[bool, Optional[float]]:
        """Returns (should_recompute, anchor_t_for_recompute)."""
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
):
    """Monkey-patch sparse_nn._build_neighbor_idx. The patch consults `state`
    and uses `compute_gradient_K_set` against `dense_model` when a refresh is
    needed. `coords_holder` is a mutable dict carrying coords_nm, coord_mask,
    residue_type for the current protein (these are set before each sampling
    pass; they're constant throughout a single trajectory)."""

    orig_build = sparse_nn._build_neighbor_idx

    def hybrid_build(self, x_t, mask, t, **kwargs):
        # x_t here is a tensor [B, n_pad, 3] (the bb_ca channel), per LocalLatentsTransformer's call signature.
        # (See proteinfoundation/nn/local_latents_transformer.py — _build_neighbor_idx receives bb_ca coords.)
        current_t = float(t.flatten()[0].item()) if isinstance(t, torch.Tensor) else float(t)
        refresh, anchor_t = state.needs_refresh(current_t)
        if refresh:
            x_t_bb = x_t  # already [B, n_pad, 3]
            mask_arg = mask  # [B, n_pad]
            # Build a synthetic x_t to anchor the K-set computation at anchor_t.
            # For frozen_t05 the anchor is 0.5; for t_grid_5 it's the nearest grid t.
            # In all cases we use the CURRENT x_t shape but evaluate dense at anchor_t.
            with torch.enable_grad():
                new_idx, new_slot_valid = compute_gradient_K_set(
                    dense_model=dense_model,
                    x_t_bb=x_t_bb,
                    mask=mask_arg,
                    t_val=anchor_t,
                    K=K_to_use,
                    coords_nm=coords_holder["coords_nm"],
                    coord_mask=coords_holder["coord_mask"],
                    residue_type=coords_holder["residue_type"],
                )
            state.cached_idx = new_idx
            state.cached_slot_valid = new_slot_valid
            state.cached_anchor_t = anchor_t
            if state.strategy == "frozen_t05":
                state.frozen = True

        state.step_counter += 1
        return state.cached_idx, state.cached_slot_valid

    sparse_nn._build_neighbor_idx = hybrid_build.__get__(sparse_nn, type(sparse_nn))
    return orig_build  # caller can restore later if needed


# ---------------------------------------------------------------------------- #
# Sampling: leverage the model's existing full_simulation, with the patched
# _build_neighbor_idx swapping in the gradient K-set.
# ---------------------------------------------------------------------------- #
def run_sampling_one(
    sparse_model: Proteina,
    coords_nm: torch.Tensor,        # [1, n_pad, 37, 3] dummy (only needed for protein metadata)
    coord_mask: torch.Tensor,       # [1, n_pad, 37]
    residue_type: torch.Tensor,     # [1, n_pad]
    mask: torch.Tensor,             # [1, n_pad]
    nsteps: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Returns the final sampled bb_ca coords [n_pad, 3]."""
    from functools import partial as _partial
    from omegaconf import OmegaConf as _OC
    torch.manual_seed(seed)
    B, n_pad = mask.shape
    assert B == 1, f"run_sampling_one is per-sample (B=1); got B={B}"

    # full_simulation squeezes any batch tensor with size(0)==1 (see
    # product_space_flow_matcher.py:626-632), which then trips the assert
    # `mask.shape == (nsamples, n)`. Workaround: pad to nsamples=2 internally,
    # return the first sample. Same per-call seed preserves paired-noise
    # semantics between the two arms at the caller level.
    B_use = 2
    mask_use = mask.expand(B_use, n_pad).contiguous()

    # Match the canonical inference path in proteina.py:predict_step. The fm
    # expects a sampling_model_args dict (the inference YAML's
    # generation.model block) and a predict_for_sampling Callable; it does NOT
    # take nn=/x_0=/additional_batch_info= directly. Pull defaults from
    # inference_base.yaml (verified 2026-05-13). CA-only path → only bb_ca.
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

    # gen_samples is a dict keyed by data_mode; CA-only path has "bb_ca".
    if isinstance(gen_samples, dict):
        if "bb_ca" in gen_samples:
            final = gen_samples["bb_ca"]
        else:
            final = next(iter(gen_samples.values()))
            if isinstance(final, dict) and "bb_ca" in final:
                final = final["bb_ca"]
    else:
        final = gen_samples

    return final[0]  # [n_pad, 3]


def save_pdb_ca_only(coords_ca: torch.Tensor, mask: torch.Tensor, residue_type: torch.Tensor, path: str):
    """Write a CA-only PDB. coords_ca in nm — convert to Å. residue_type is the
    OpenFold residue-index tensor.
    """
    # CA-only writer — minimal PDB with CA atoms only.
    # AA 3-letter codes by residue_type index. La-Proteina uses OpenFold ordering.
    AA_THREE = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK",
    ]
    coords_ang = coords_ca.detach().cpu().numpy() * 10.0  # nm → Å
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
        "--routing_strategy",
        choices=["frozen_t05", "t_grid_5", "every_M"],
        default="frozen_t05",
        help="K-set update strategy. frozen_t05 is cheapest (~30-60 min total). "
             "t_grid_5 (~1-2h) and every_M (~3-4h) refresh more often.",
    )
    parser.add_argument("--M_steps", type=int, default=50,
                        help="For every_M strategy: refresh K-set every M ODE steps.")
    parser.add_argument("--nsteps", type=int, default=400,
                        help="ODE steps. CLAUDE.md hard rule: do not override down for "
                             "designability evals.")
    parser.add_argument("--seed", type=int, default=5,
                        help="Matches recent N=6 probe seeds; use a fresh seed if "
                             "pooling per feedback_seed_propagation_audit.")
    parser.add_argument(
        "--baseline_only", action="store_true",
        help="Run only the §11 baseline arm (no gradient routing).",
    )
    parser.add_argument(
        "--hybrid_only", action="store_true",
        help="Run only the hybrid arm (skip baseline).",
    )
    parser.add_argument("--out_dir", default="results/hybrid_grad_routing")
    args = parser.parse_args()

    load_dotenv()
    L.seed_everything(args.seed)
    assert args.nsteps == 400, (
        "nsteps must be 400 for designability per CLAUDE.md hard rule. "
        "Override only for diagnostic runs that are NOT evaluated with scRMSD."
    )

    lengths = [int(x) for x in args.lengths.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Gradient routing requires CUDA for the backward pass."

    out_root = Path(args.out_dir) / args.label
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 80)
    logger.info(f"Hybrid gradient-routing inference | label={args.label}")
    logger.info(f"  dense_ckpt:  {args.dense_ckpt}")
    logger.info(f"  sparse_ckpt: {args.sparse_ckpt}")
    logger.info(f"  N={args.n_samples} × L={lengths} × nsteps={args.nsteps}")
    logger.info(f"  routing_strategy={args.routing_strategy}, K={args.K}, seed={args.seed}")
    logger.info(f"  arms: baseline={not args.hybrid_only}, hybrid={not args.baseline_only}")
    logger.info("=" * 80)

    # ------------------- Load models -------------------
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

    # Unwrap torch.compile: the ckpt's `cfg_exp.opt.compile_nn=True` makes
    # proteina.py wrap `self.nn` with torch.compile at init. Dynamo cannot
    # trace the patched `_build_neighbor_idx` (it sees K as a SymInt and
    # einops.rearrange raises `unhashable type: non-nested SymInt`). Swap
    # back to the original module before patching. Slower per forward but
    # correct for this hybrid path.
    if hasattr(sparse_model.nn, "_orig_mod"):
        logger.info("  unwrapping sparse_model.nn from torch.compile (hybrid patch is incompatible with dynamo).")
        sparse_model.nn = sparse_model.nn._orig_mod
    if hasattr(dense_model.nn, "_orig_mod"):
        logger.info("  unwrapping dense_model.nn from torch.compile (per-query VJP needs dense without dynamo).")
        dense_model.nn = dense_model.nn._orig_mod

    # Confirm sparse_attention flag
    sp_cfg = sparse_model.cfg_exp.get("nn", {})
    assert sp_cfg.get("sparse_attention", False), (
        f"sparse_ckpt at {args.sparse_ckpt} has sparse_attention=False. Wrong ckpt?"
    )
    de_cfg = dense_model.cfg_exp.get("nn", {})
    assert not de_cfg.get("sparse_attention", False), (
        f"dense_ckpt at {args.dense_ckpt} has sparse_attention=True. Wrong ckpt?"
    )

    n_layers = len(sparse_model.nn.transformer_layers)
    logger.info(f"  sparse: n_layers={n_layers}, K_seq+sp+rd={sp_cfg.get('n_seq_neighbors')}*2 "
                f"+ {sp_cfg.get('n_spatial_neighbors')} + {sp_cfg.get('n_random_neighbors')}")

    # ------------------- Set up the routing state + patch -------------------
    state = HybridRoutingState(
        strategy=args.routing_strategy,
        t_grid=T_GRID_5_DEFAULT,
        M_steps=args.M_steps,
    )
    coords_holder: Dict[str, torch.Tensor] = {}  # filled per protein

    # Save original _build_neighbor_idx for restoration / baseline runs
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
        )

    def uninstall_hybrid_patch():
        sparse_model.nn._build_neighbor_idx = original_build

    # ------------------- Per (length, sample) loop -------------------
    metadata: List[dict] = []
    for L_target in lengths:
        n_pad = max(L_target + 16, 64)  # small headroom; rounded up
        for prot_idx in range(args.n_samples):
            seed = args.seed + prot_idx * 1000 + L_target
            logger.info(f"--- L={L_target} sample {prot_idx} (seed={seed}) ---")

            # Build a synthetic protein record: residue_type all ALA (=0),
            # coords_nm all zero (real coords aren't needed for from-noise sampling).
            coords_nm = torch.zeros(1, n_pad, 37, 3, device=device)
            coord_mask = torch.zeros(1, n_pad, 37, dtype=torch.bool, device=device)
            coord_mask[0, :L_target, 1] = True  # CA atom present
            residue_type = torch.zeros(1, n_pad, dtype=torch.long, device=device)
            mask = torch.zeros(1, n_pad, dtype=torch.bool, device=device)
            mask[0, :L_target] = True

            coords_holder["coords_nm"] = coords_nm
            coords_holder["coord_mask"] = coord_mask
            coords_holder["residue_type"] = residue_type

            # ------ Baseline arm: vanilla §11 (default K-set) ------
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

            # ------ Hybrid arm: §11 with gradient-derived K-set ------
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
                            f"K-set refreshes={state.step_counter // max(1, state.M_steps) if state.strategy.startswith('every_') else 'see strategy'})")
                metadata.append({
                    "arm": "hybrid",
                    "L": L_target,
                    "sample_idx": prot_idx,
                    "seed": seed,
                    "routing_strategy": args.routing_strategy,
                    "K": args.K,
                    "path": str(hpath),
                    "wall_s": t_h,
                })
                # Restore baseline build for the next protein's baseline arm
                uninstall_hybrid_patch()

    # ------------------- Write run metadata -------------------
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
                "seed": args.seed,
                "samples": metadata,
            },
            f,
            indent=2,
        )
    logger.info(f"Metadata written to {meta_path}")
    logger.info("Done. Next step: run evaluate.py / scRMSD on the saved PDBs.")


if __name__ == "__main__":
    main()
