"""Dense self-sparsification: test whether dense survives K=64 attention constraint.

The question this answers (decision-gate for the 15h routing-distillation
training slot):
  - If dense self-survives the K=64 mask → K=64 budget is sufficient; the
    routing-prior bottleneck is the ONLY barrier between sparse and dense.
    Training-side fix (E063 follow-up — train sparse with attention K-set
    teacher) is well-motivated.
  - If dense self-fails the K=64 mask → K=64 is fundamentally too few even
    for dense's own weights; sparse will never match dense regardless of
    routing quality. Save the 15h slot; move to other axes (per-layer random
    redraw, length-scaled K, or accept dense at N ≤ 800).

Mechanism (FROZEN strategy, quickest version):
  - For each protein, sample initial noise x_0 and run a SINGLE unmasked dense
    forward at t=0 to capture dense's softmax attention at (layer=1, head=7).
    Take top-K per query as a frozen K-set for this protein.
  - Patch PairBiasAttention._attn to apply that K-set as a pre-softmax mask
    at EVERY layer × head: each query i can only attend to its own top-K
    residues (selected from L1 H7's pattern at t=0). All other attention
    scores set to -inf before softmax.
  - Run dense's standard sampling at nsteps=400. All 400 ODE steps use the
    same frozen K-set per query.
  - Baseline arm: vanilla dense (no masking) on the same paired noise.

Both arms use dense's weights — this is NOT a sparse vs dense comparison.
It's vanilla-dense vs K=64-attention-budget-constrained-dense.

Quick mode: N=3 × L ∈ {50, 100} × nsteps=400. ~30-45 min on 1× A100.
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
from proteinfoundation.nn.modules.pair_bias_attn import PairBiasAttention


DENSE_CKPT_DEFAULT = (
    "/rds/user/ks2218/hpc-work/store/test_ca_only_diffusion/1776805213/"
    "checkpoints/best_val_00000026_000000002646.ckpt"
)

# E061 top winner — L1 H7 = 18.3% of queries' best-match (l, h) vs gradient.
DEFAULT_CAPTURE_LAYER = 1
DEFAULT_CAPTURE_HEAD = 7


# ---------------------------------------------------------------------------- #
# Global state for the patched _attn. Holds the K-set (set during capture
# phase) and a mode flag (capture vs apply mask).
# ---------------------------------------------------------------------------- #
class SelfSparsifyState:
    def __init__(self):
        self.K_set_per_query: Optional[torch.Tensor] = None  # [N_real, K]
        self.mask_mode: bool = False
        self.capture_target_layer: int = DEFAULT_CAPTURE_LAYER
        self.capture_target_head: int = DEFAULT_CAPTURE_HEAD
        self.N_real: int = 0
        self.K: int = 64

    def reset(self, N_real: int, K: int, target_layer: int, target_head: int):
        self.K_set_per_query = None
        self.mask_mode = False
        self.N_real = N_real
        self.K = K
        self.capture_target_layer = target_layer
        self.capture_target_head = target_head


STATE = SelfSparsifyState()


def make_dense_sparsify_attn():
    """Class-wide replacement for PairBiasAttention._attn (dense path).

    Behavior depends on STATE.mask_mode:
      - mask_mode=False (capture phase): standard dense attention, BUT record
        top-K per query at (target_layer, target_head) into STATE.K_set_per_query.
      - mask_mode=True (sample phase): apply pre-softmax mask using
        STATE.K_set_per_query — each real query i can only attend to keys in
        K_set_per_query[i]. Padding queries unaffected (their attention
        doesn't matter; mask preserves their pad-mask behavior).
    """
    from einops import rearrange
    from torch import einsum

    max_neg_value = lambda x: torch.finfo(x.dtype).min  # noqa: E731

    def patched_attn(self_attn, q, k, v, b, mask_arg):
        # Standard QK^T + scale + pair-mask, as in PairBiasAttention._attn.
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self_attn.scale
        if mask_arg is not None:
            mask_rs = rearrange(mask_arg, "b i j -> b () i j")
            sim = sim.masked_fill(~mask_rs, max_neg_value(sim))

        # Apply K-set mask BEFORE softmax (in mask_mode).
        if STATE.mask_mode and STATE.K_set_per_query is not None:
            B, H, N, _ = sim.shape
            N_real = STATE.N_real
            K = STATE.K
            # keep_mask[i, j] = True if j ∈ K_set_per_query[i], for real queries i.
            keep_mask = torch.zeros(N_real, N, dtype=torch.bool, device=sim.device)
            keep_mask.scatter_(1, STATE.K_set_per_query, True)
            keep_full = keep_mask.unsqueeze(0).unsqueeze(0).expand(B, H, N_real, N)
            # Padding queries (rows ≥ N_real): keep all (they're masked out by
            # the protein mask anyway; their attention pattern is downstream
            # of irrelevant padding values).
            pad_keep = torch.ones(
                B, H, N - N_real, N, dtype=torch.bool, device=sim.device
            )
            keep_combined = torch.cat([keep_full, pad_keep], dim=2)
            sim = sim.masked_fill(~keep_combined, max_neg_value(sim))

        attn = torch.softmax(sim + b, dim=-1).nan_to_num(0.0)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # In capture mode, snapshot top-K at the target (layer, head).
        if not STATE.mask_mode:
            layer_idx = getattr(self_attn, "_layer_idx", -1)
            if layer_idx == STATE.capture_target_layer:
                attn_lh = attn[
                    0, STATE.capture_target_head, : STATE.N_real, : STATE.N_real
                ].float()
                # Mask invalid keys (already taken care of by mask_arg, but safe).
                Keff = min(STATE.K, STATE.N_real)
                top_K = attn_lh.topk(Keff, dim=-1).indices  # [N_real, Keff]
                if Keff < STATE.K:
                    pad_idx = torch.zeros(
                        STATE.N_real, STATE.K - Keff, device=sim.device, dtype=torch.long
                    )
                    top_K = torch.cat([top_K, pad_idx], dim=-1)
                STATE.K_set_per_query = top_K.detach()

        return out

    return patched_attn


# ---------------------------------------------------------------------------- #
# Capture phase: one unmasked dense forward at x_t = noise to populate
# STATE.K_set_per_query.
# ---------------------------------------------------------------------------- #
@torch.no_grad()
def capture_K_set(
    dense_model: Proteina,
    coords_nm: torch.Tensor,
    coord_mask: torch.Tensor,
    residue_type: torch.Tensor,
    mask: torch.Tensor,
    t_val: float,
    device: torch.device,
):
    """Run ONE unmasked dense forward to capture L1 H7's top-K per query.
    Sets STATE.K_set_per_query. Called once before sampling for FROZEN mode.
    """
    B, n_pad = mask.shape
    # x_t = pure noise (t=0). We don't have a target structure, so we just
    # sample fresh noise — the captured K-set is then "what L1 H7 attends to
    # when given a noise input at t=0," which is the canonical low-t routing
    # prior. (Frozen at this; doesn't update during sampling.)
    x_t_bb = torch.randn(B, n_pad, 3, device=device)

    STATE.mask_mode = False  # capture mode
    STATE.K_set_per_query = None

    batch = {
        "coords_nm": coords_nm,
        "coord_mask": coord_mask,
        "residue_type": residue_type,
        "mask": mask,
        "x_t": {"bb_ca": x_t_bb},
        "t": {"bb_ca": torch.full((B,), float(t_val), device=device)},
    }
    # Tag layers (do this once per call, idempotent).
    for i, layer in enumerate(dense_model.nn.transformer_layers):
        layer.mhba.mha._layer_idx = i

    _ = dense_model.call_nn(batch, n_recycle=0)
    assert STATE.K_set_per_query is not None, (
        f"Capture failed: target_layer={STATE.capture_target_layer} not seen"
    )


# ---------------------------------------------------------------------------- #
# Sampling (same predict_for_sampling pattern as the hybrid sibling scripts).
# ---------------------------------------------------------------------------- #
def run_sampling_one(
    dense_model: Proteina,
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

    # B=2 pad workaround for full_simulation's squeeze+assert bug
    # (product_space_flow_matcher.py:626-632).
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
    batch = {"nsamples": B_use, "nres": n_pad, "mask": mask_use}
    fn_predict_for_sampling = _partial(
        dense_model.predict_for_sampling, n_recycle=0
    )
    sc_neighbors_active = dense_model.cfg_exp.training.get("sc_neighbors", False)
    with torch.no_grad():
        gen_samples, _info = dense_model.fm.full_simulation(
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


def save_pdb_ca_only(coords_ca: torch.Tensor, mask: torch.Tensor,
                     residue_type: torch.Tensor, path: str):
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
    parser.add_argument(
        "--n_samples", type=int, default=3,
        help="N per length. Quick default: 3.",
    )
    parser.add_argument(
        "--lengths", type=str, default="50,100",
        help="Quick default skips L=200 (where everyone dies anyway).",
    )
    parser.add_argument(
        "--K", type=int, default=64,
        help="Attention K-set size per query (matches §11's K).",
    )
    parser.add_argument(
        "--capture_layer", type=int, default=DEFAULT_CAPTURE_LAYER,
        help=f"Dense layer to capture K-set from. Default {DEFAULT_CAPTURE_LAYER} "
             f"(L{DEFAULT_CAPTURE_LAYER} H{DEFAULT_CAPTURE_HEAD} = E061's top winner).",
    )
    parser.add_argument(
        "--capture_head", type=int, default=DEFAULT_CAPTURE_HEAD,
        help="Head within --capture_layer.",
    )
    parser.add_argument("--nsteps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--baseline_only", action="store_true")
    parser.add_argument("--hybrid_only", action="store_true")
    parser.add_argument("--out_dir", default="results/dense_self_sparsify")
    args = parser.parse_args()

    load_dotenv()
    L.seed_everything(args.seed)
    assert args.nsteps == 400, "nsteps must be 400 per CLAUDE.md hard rule."

    lengths = [int(x) for x in args.lengths.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Inference requires CUDA."

    out_root = Path(args.out_dir) / args.label
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 80)
    logger.info(f"Dense self-sparsification | label={args.label}")
    logger.info(f"  dense_ckpt: {args.dense_ckpt}")
    logger.info(f"  N={args.n_samples} × L={lengths} × nsteps={args.nsteps}")
    logger.info(f"  K={args.K}, capture (L{args.capture_layer} H{args.capture_head})")
    logger.info(f"  arms: baseline={not args.hybrid_only}, hybrid={not args.baseline_only}")
    logger.info("=" * 80)

    logger.info("Loading dense model...")
    dense_model = Proteina.load_from_checkpoint(
        args.dense_ckpt, strict=False, autoencoder_ckpt_path=None
    )
    dense_model.to(device).eval()
    for p in dense_model.parameters():
        p.requires_grad_(False)

    if hasattr(dense_model.nn, "_orig_mod"):
        logger.info("  unwrapping dense_model.nn from torch.compile.")
        dense_model.nn = dense_model.nn._orig_mod

    de_cfg = dense_model.cfg_exp.get("nn", {})
    assert not de_cfg.get("sparse_attention", False), (
        f"dense_ckpt has sparse_attention=True. Wrong ckpt?"
    )
    n_layers_dense = len(dense_model.nn.transformer_layers)
    n_heads_dense = de_cfg.get("nheads", 12)
    assert 0 <= args.capture_layer < n_layers_dense
    assert 0 <= args.capture_head < n_heads_dense

    # Save original _attn for the baseline arm.
    original_attn = PairBiasAttention._attn
    patched_attn = make_dense_sparsify_attn()

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

            # ----- Baseline: vanilla dense -----
            if not args.hybrid_only:
                PairBiasAttention._attn = original_attn  # unpatched
                STATE.mask_mode = False
                STATE.K_set_per_query = None
                t0 = time.time()
                final_baseline = run_sampling_one(
                    dense_model=dense_model,
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

            # ----- Hybrid: dense with attention sparsified to L1 H7 top-K -----
            if not args.baseline_only:
                # Phase 1: capture K-set from L1 H7 at x_t = noise.
                STATE.reset(
                    N_real=L_target,
                    K=args.K,
                    target_layer=args.capture_layer,
                    target_head=args.capture_head,
                )
                PairBiasAttention._attn = patched_attn  # capture mode (mask_mode=False)
                torch.manual_seed(seed)  # paired with baseline's sampling seed
                capture_K_set(
                    dense_model=dense_model,
                    coords_nm=coords_nm,
                    coord_mask=coord_mask,
                    residue_type=residue_type,
                    mask=mask,
                    t_val=0.0,
                    device=device,
                )
                logger.info(
                    f"  captured K-set: shape={tuple(STATE.K_set_per_query.shape)}, "
                    f"min_idx={int(STATE.K_set_per_query.min())}, "
                    f"max_idx={int(STATE.K_set_per_query.max())}"
                )

                # Phase 2: sampling with mask_mode=True (apply stored K-set).
                STATE.mask_mode = True
                t0 = time.time()
                final_hybrid = run_sampling_one(
                    dense_model=dense_model,
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
                logger.info(f"  hybrid   saved {hpath}  ({t_h:.1f}s)")
                metadata.append({
                    "arm": "hybrid",
                    "L": L_target,
                    "sample_idx": prot_idx,
                    "seed": seed,
                    "K": args.K,
                    "capture_layer": args.capture_layer,
                    "capture_head": args.capture_head,
                    "path": str(hpath),
                    "wall_s": t_h,
                })
                # Restore original _attn for the next baseline run.
                PairBiasAttention._attn = original_attn
                STATE.mask_mode = False
                STATE.K_set_per_query = None

    meta_path = out_root / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "label": args.label,
                "dense_ckpt": args.dense_ckpt,
                "K": args.K,
                "nsteps": args.nsteps,
                "n_samples": args.n_samples,
                "lengths": lengths,
                "capture_layer": args.capture_layer,
                "capture_head": args.capture_head,
                "seed": args.seed,
                "samples": metadata,
            },
            f, indent=2,
        )
    logger.info(f"Metadata written to {meta_path}")
    logger.info("Done. Next: run eval_hybrid_grad_routing.py or evaluate.py on the saved PDBs.")


if __name__ == "__main__":
    main()
