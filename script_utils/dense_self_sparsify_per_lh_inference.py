"""Dense self-sparsification — PER-(layer, head) top-K variant.

Sibling of `dense_self_sparsify_inference.py` (E063 — single shared K-set
captured at L1 H7 t=0 and enforced on all 14 × 12 cells). The user's
follow-up question was: does the L=100 catastrophe in that experiment come
from "K=64 is fundamentally too few for dense's own weights" or from
"forcing L1 H7's K-set on every (layer, head) cell that has its own
specialization"?

This variant tests the cleaner version: **each (layer, head) at each ODE
step computes its OWN top-K=64 attended residues from its OWN softmax
scores**, on the fly. No capture phase; no shared K-set; no freezing.

Mechanism (inside the patched PairBiasAttention._attn):
  1. Standard sim = Q · K^T * scale, with pair-mask applied as in the
     original.
  2. Add pair bias b to sim to get the full pre-softmax score.
  3. For each (batch, head, query) cell, take top-K over keys and zero
     out the rest (set to -inf before softmax). Real-key masking is
     enforced explicitly so padding keys can never enter top-K.
  4. softmax normalises over the K kept keys per (B, H, query).

Both arms:
  - baseline: vanilla dense (full N × N attention), unpatched _attn.
  - hybrid:   dense weights run identically, but every layer × head's
              softmax distribution is restricted to its own top-K per query
              at every ODE step.

Paired-by-noise A/B vs the same seeds the E063 shared-K-set version used,
so the two experiments are comparable cell-by-cell.

Quick mode (default): N=3 × L ∈ {50, 100} × nsteps=400. ~3-5 min on 1× L4.

Decision-tree disambiguation vs E063:
  - hybrid pool ≈ baseline pool (e.g. 5-6/6 vs 5-6/6)  → K=64 IS sufficient
    for dense when each cell picks its own K-set. The E063 catastrophe was
    "wrong K-set enforced everywhere", not "K=64 fundamentally too few".
    Training-side per-query routing with per-(layer, head) heads is well-
    motivated.
  - hybrid pool ≈ E063 hybrid pool (both bad) → K=64 IS fundamentally too
    few for dense even with per-cell freedom; the catastrophe in E063 was
    real K-sufficiency, not shared-K-set artefact.
  - intermediate → mixed read; per-cell K helps but doesn't fully recover.
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


# ---------------------------------------------------------------------------- #
# Global state — just K and N_real (the per-(layer, head) top-K happens
# entirely inside the patched _attn; no captured tensor needed).
# ---------------------------------------------------------------------------- #
class PerLHState:
    def __init__(self):
        self.enabled: bool = False
        self.N_real: int = 0
        self.K: int = 64
        # t-gated masking: only apply the top-K mask when current_t >= mask_t_start.
        # mask_t_start = 0.0 → always mask (default; matches the original
        # per-(L,H) experiment). mask_t_start = 0.5 → unmasked attention from
        # t=0 (noise) to t=0.5, then top-K mask from t=0.5 to t=1 (clean).
        self.current_t: float = 0.0
        self.mask_t_start: float = 0.0
        self.mask_t_end: float = 1.1   # default: never blocks (current_t < 1.1 always true)
        # If True: zero non-top-K positions in the post-softmax attention (no renorm).
        # If False (default): mask pre-softmax to -inf and let softmax renormalize.
        self.no_renormalize: bool = False
        # If True: keep RANDOM K residues per (B, H, query) cell instead of
        # the top-K by attention score. Control experiment for whether the
        # model's own attention pattern is the informative part of the mask.
        self.random_keep: bool = False
        # Diagnostic counters.
        self.n_calls: int = 0
        self.n_masks_fired: int = 0
        self.t_values_seen: List[float] = []

    def reset(self, N_real: int, K: int, enabled: bool,
              mask_t_start: float = 0.0, mask_t_end: float = 1.1,
              no_renormalize: bool = False, random_keep: bool = False):
        self.N_real = N_real
        self.K = K
        self.enabled = enabled
        self.mask_t_start = mask_t_start
        self.mask_t_end = mask_t_end
        self.no_renormalize = no_renormalize
        self.random_keep = random_keep
        self.current_t = 0.0
        self.n_calls = 0
        self.n_masks_fired = 0
        self.t_values_seen = []


STATE = PerLHState()


def make_per_lh_topk_attn():
    """Class-wide replacement for PairBiasAttention._attn.

    Behavior depends on STATE.enabled:
      - enabled=False (baseline): standard dense attention, unchanged.
      - enabled=True  (hybrid):   each (B, H, query) row of the pre-softmax
                                  score matrix is top-K-masked over keys
                                  before softmax.

    Note: pair bias `b` is included in the score used for top-K selection
    (the original code applies `softmax(sim + b)`, so the pre-softmax
    score is `sim + b`). This matches "what dense actually attends to".
    """
    from einops import rearrange
    from torch import einsum

    max_neg_value = lambda x: torch.finfo(x.dtype).min  # noqa: E731

    def patched_attn(self_attn, q, k, v, b, mask_arg):
        # Standard QK^T + scale.
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self_attn.scale

        # Pair-mask (same as original).
        if mask_arg is not None:
            mask_rs = rearrange(mask_arg, "b i j -> b () i j")
            sim = sim.masked_fill(~mask_rs, max_neg_value(sim))

        # Full pre-softmax score, before top-K masking.
        score = sim + b

        # t-gated: only fire when current trajectory t >= mask_t_start. Lets
        # us test "mask only the second half (t>=0.5) of the trajectory" to
        # see if the L=200 collapse is driven by t=0 randomness or by the
        # whole trajectory.
        if STATE.enabled:
            STATE.n_calls += 1
        if (STATE.enabled
                and STATE.current_t >= STATE.mask_t_start
                and STATE.current_t < STATE.mask_t_end):
            STATE.n_masks_fired += 1
            # Only sample a few t values to avoid log explosion.
            if STATE.n_masks_fired <= 5 or STATE.n_masks_fired % 200 == 0:
                STATE.t_values_seen.append(STATE.current_t)
            B, H, N, _ = score.shape
            N_real = STATE.N_real
            K = STATE.K
            Keff = min(K, N_real)

            # Mask invalid (padding) keys to -inf so they cannot enter top-K.
            real_keys_mask = torch.zeros(N, dtype=torch.bool, device=score.device)
            real_keys_mask[:N_real] = True
            score_for_topk = score.masked_fill(
                ~real_keys_mask.view(1, 1, 1, N), max_neg_value(score)
            )

            if STATE.random_keep:
                # Random K per (B, H, query): independent random permutation
                # of real keys per cell, take first Keff. Drives the score
                # for non-kept real keys to -inf via a random projection.
                # Padding keys remain -inf via real_keys_mask.
                rand = torch.rand(B, H, N, N_real, device=score.device)
                # First Keff indices of randperm-by-argsort.
                top_K_idx = rand.argsort(dim=-1)[..., :Keff]  # [B, H, N, Keff], values in [0, N_real)
            else:
                # Per-(B, H, query) top-K over keys by attention score.
                top_K_idx = score_for_topk.topk(k=Keff, dim=-1).indices  # [B, H, N, Keff]

            # Build keep_mask of shape [B, H, N, N].
            keep_mask = torch.zeros_like(score, dtype=torch.bool)
            keep_mask.scatter_(-1, top_K_idx, True)
            # Padding rows (i >= N_real): keep all (their attention pattern is
            # downstream of padding-masked outputs anyway).
            keep_mask[..., N_real:, :] = True

            if STATE.no_renormalize:
                # Compute the natural softmax (no truncation), then zero non-top-K.
                # Total mass per query shrinks from 1.0 to ~mass_top_K; kept weights
                # retain their unscaled values (no 1.21× inflation per kept key).
                attn = torch.softmax(score, dim=-1).nan_to_num(0.0)
                attn = attn * keep_mask.to(attn.dtype)
                out = einsum("b h i j, b h j d -> b h i d", attn, v)
                return out
            else:
                # Original behavior: pre-softmax mask → softmax renormalizes
                # the kept keys to sum to 1.0 (causes ~+20% per-key inflation
                # at L=200, K=64).
                score = score.masked_fill(~keep_mask, max_neg_value(score))

        attn = torch.softmax(score, dim=-1).nan_to_num(0.0)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        return out

    return patched_attn


# ---------------------------------------------------------------------------- #
# Sampling: canonical-style call (single B=N batch, seed_everything once per
# batch, natural noise tensor with no B=1+pad workaround). This matches what
# proteina.py:predict_step does and what `diagnose_dense_baseline.py` verified
# reproduces the 5/6 designable canonical baseline.
# ---------------------------------------------------------------------------- #
def run_sampling_batch(
    dense_model: Proteina,
    mask: torch.Tensor,            # [B, n_pad] bool, all rows same protein length
    nsteps: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    from functools import partial as _partial
    from omegaconf import OmegaConf as _OC

    L.seed_everything(seed)
    B, n_pad = mask.shape
    assert B >= 2, f"full_simulation squeeze+assert needs B>=2; got B={B}"

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
    batch = {"nsamples": B, "nres": n_pad, "mask": mask}
    # Wrap predict_for_sampling to thread the current trajectory t into
    # STATE.current_t. The patched _attn reads STATE.current_t for the t-gate.
    base_predict = _partial(dense_model.predict_for_sampling, n_recycle=0)
    def fn_predict_for_sampling(batch_dict, *args_, **kwargs_):
        t_val = batch_dict.get("t", {})
        if isinstance(t_val, dict) and "bb_ca" in t_val:
            t_tensor = t_val["bb_ca"]
            if hasattr(t_tensor, "flatten"):
                STATE.current_t = float(t_tensor.flatten()[0].item())
        return base_predict(batch_dict, *args_, **kwargs_)
    sc_neighbors_active = dense_model.cfg_exp.training.get("sc_neighbors", False)
    with torch.no_grad():
        gen_samples, _info = dense_model.fm.full_simulation(
            batch=batch,
            predict_for_sampling=fn_predict_for_sampling,
            nsteps=nsteps,
            nsamples=B,
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
    return final  # [B, n_pad, 3]


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
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument(
        "--lengths", type=str, default="50,100",
        help="Quick default skips L=200.",
    )
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated list of seed bases (overrides --seed). Each seed "
             "produces N samples per length; PDBs saved at global sample_idx "
             "= seed_idx * N + s.",
    )
    parser.add_argument(
        "--mask_t_start", type=float, default=0.0,
        help="t threshold (lower bound): mask fires only when current_t >= "
             "this value. 0.0 = always mask (default). 0.5 = unmasked for "
             "t in [0, 0.5), masked for t in [0.5, 1]. La-Proteina "
             "convention: t=0 noise, t=1 clean.",
    )
    parser.add_argument(
        "--random_keep", action="store_true",
        help="Control: keep K RANDOM keys per (layer, head, query) instead "
             "of the top-K by attention score. Tests whether dense's own "
             "attention pattern is the informative part of the mask.",
    )
    parser.add_argument(
        "--no_renormalize", action="store_true",
        help="If set: keep the top-K attention weights at their original (unscaled) "
             "softmax values; the dropped weights become zero. Total per-query "
             "attention mass drops from 1.0 to ~mass_top_K instead of staying at "
             "1.0 with the kept keys inflated by 1/mass_top_K. Tests whether the "
             "renormalization is the destructive part vs the mass loss.",
    )
    parser.add_argument(
        "--mask_t_end", type=float, default=1.1,
        help="t threshold (upper bound): mask fires only when current_t < "
             "this value. 1.1 = effectively no upper bound (default). "
             "Combined: mask fires when current_t ∈ [mask_t_start, mask_t_end). "
             "Use --mask_t_start 0.0 --mask_t_end 0.5 to mask ONLY the noisy "
             "early-trajectory half (test of 'random top-K at low-t' hypothesis).",
    )
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
    logger.info(f"Dense self-sparsification PER-(layer, head) | label={args.label}")
    logger.info(f"  dense_ckpt: {args.dense_ckpt}")
    logger.info(f"  N={args.n_samples} × L={lengths} × nsteps={args.nsteps}")
    logger.info(f"  K={args.K} per (layer, head, query), recomputed every ODE step from softmax(QK^T + b)")
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

    original_attn = PairBiasAttention._attn
    patched_attn = make_per_lh_topk_attn()

    metadata: List[dict] = []
    N = args.n_samples
    if N < 2:
        raise ValueError(f"n_samples must be ≥ 2 (got {N}) due to full_simulation's squeeze+assert.")

    # Build the seed base list. Default: just [args.seed]. With --seeds: parse.
    if args.seeds is not None:
        seed_bases = [int(x) for x in args.seeds.split(",")]
    else:
        seed_bases = [args.seed]
    logger.info(f"  seed bases: {seed_bases}  → {len(seed_bases)} × N={N} = {len(seed_bases) * N} samples per (arm, L)")

    for seed_idx, seed_base in enumerate(seed_bases):
        for batch_idx, L_target in enumerate(lengths):
            n_pad = max(L_target + 16, 64)
            # Canonical seed scheme within each seed_base: seed_for_L = seed_base + batch_idx,
            # matching predict_step in proteina.py:809-811.
            seed_for_L = seed_base + batch_idx

            mask = torch.zeros(N, n_pad, dtype=torch.bool, device=device)
            mask[:, :L_target] = True
            residue_type = torch.zeros(L_target, dtype=torch.long)

            logger.info(f"=== seed_idx={seed_idx} (base={seed_base}) L={L_target} batch_idx={batch_idx} seed={seed_for_L} N={N} n_pad={n_pad} ===")

            # Global sample-index offset: seed_idx * N + s. Lets all results
            # land flat in <label>/<arm>/L<N>/sample_{0..len(seeds)*N-1}.pdb.
            idx_offset = seed_idx * N

            # ----- Baseline: vanilla dense -----
            if not args.hybrid_only:
                PairBiasAttention._attn = original_attn
                STATE.enabled = False
                t0 = time.time()
                final_baseline = run_sampling_batch(
                    dense_model=dense_model,
                    mask=mask,
                    nsteps=args.nsteps,
                    seed=seed_for_L,
                    device=device,
                )  # [N, n_pad, 3]
                t_b = time.time() - t0
                bdir = out_root / "baseline" / f"L{L_target}"
                bdir.mkdir(parents=True, exist_ok=True)
                for s in range(N):
                    bpath = bdir / f"sample_{idx_offset + s}.pdb"
                    save_pdb_ca_only(
                        final_baseline[s, :L_target], mask[s, :L_target],
                        residue_type, str(bpath)
                    )
                    metadata.append({
                        "arm": "baseline",
                        "L": L_target,
                        "sample_idx": idx_offset + s,
                        "seed_base": seed_base,
                        "seed": seed_for_L,
                        "path": str(bpath),
                    })
                logger.info(f"  baseline saved {N} PDBs to {bdir} (indices {idx_offset}..{idx_offset+N-1})  ({t_b:.1f}s)")

            # ----- Hybrid: dense with per-(layer, head) top-K mask -----
            if not args.baseline_only:
                STATE.reset(N_real=L_target, K=args.K, enabled=True,
                            mask_t_start=args.mask_t_start, mask_t_end=args.mask_t_end,
                            no_renormalize=args.no_renormalize,
                            random_keep=args.random_keep)
                PairBiasAttention._attn = patched_attn
                t0 = time.time()
                final_hybrid = run_sampling_batch(
                    dense_model=dense_model,
                    mask=mask,
                    nsteps=args.nsteps,
                    seed=seed_for_L,  # same seed as baseline → paired noise
                    device=device,
                )
                t_h = time.time() - t0
                hdir = out_root / "hybrid" / f"L{L_target}"
                hdir.mkdir(parents=True, exist_ok=True)
                for s in range(N):
                    hpath = hdir / f"sample_{idx_offset + s}.pdb"
                    save_pdb_ca_only(
                        final_hybrid[s, :L_target], mask[s, :L_target],
                        residue_type, str(hpath)
                    )
                    metadata.append({
                        "arm": "hybrid",
                        "L": L_target,
                        "sample_idx": idx_offset + s,
                        "seed_base": seed_base,
                        "seed": seed_for_L,
                        "K": args.K,
                        "strategy": "per_layer_head_topK_every_step",
                        "path": str(hpath),
                    })
                logger.info(f"  hybrid   saved {N} PDBs to {hdir} (indices {idx_offset}..{idx_offset+N-1})  ({t_h:.1f}s, per-(L,H) top-K={args.K})")
                # Restore for the next baseline.
                PairBiasAttention._attn = original_attn
                STATE.enabled = False

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
                "strategy": "per_layer_head_topK_every_step",
                "seed": args.seed,
                "samples": metadata,
            },
            f, indent=2,
        )
    logger.info(f"Metadata written to {meta_path}")
    logger.info(
        f"t-gating diagnostic: STATE.n_calls={STATE.n_calls}, "
        f"STATE.n_masks_fired={STATE.n_masks_fired}, "
        f"mask_t_start={STATE.mask_t_start}, "
        f"sampled t-values where mask fired (first 5 + every 200th): {STATE.t_values_seen[:20]}"
    )
    logger.info("Done. Next: run eval_hybrid_grad_routing.py on the saved PDBs.")


if __name__ == "__main__":
    main()
