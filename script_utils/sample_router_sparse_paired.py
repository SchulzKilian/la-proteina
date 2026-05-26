"""Move-2 paired sampling: router-sparse trunk vs canonical dense baseline.

For each protein index in [0, n_samples), each length in --lengths, we:
  1. Compute a deterministic seed = base_seed + 1000*sample_idx + L.
  2. Sample one protein from the SPARSE trunk ckpt at that seed.
  3. Sample one protein from the DENSE baseline at that seed.

The same seed → identical x_0 noise per the rdn_flow_matcher path; the only
difference is the model. This is the same pattern as dense_self_sparsify_inference.py.

Output PDB tree:
  <out_root>/<label>/sparse/L{L}/sample_{i}.pdb
  <out_root>/<label>/baseline/L{L}/sample_{i}.pdb

Eval (paired ΔscRMSD) is done separately by eval_router_sparse_paired.py.

Use this for both the full eval (N=12 per L) AND the intermediate-checkpoint
designability probes (N=6 per L) — vary --n_samples.
"""
import argparse
import json
import os
import sys
import time
from functools import partial as _partial
from pathlib import Path
from typing import List

REPO = "/home/ks2218/la-proteina"
sys.path.insert(0, REPO)

import lightning as L
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf as _OC

from proteinfoundation.proteina import Proteina


DENSE_CKPT_DEFAULT = (
    "/rds/user/ks2218/hpc-work/store/test_ca_only_diffusion/1776805213/"
    "checkpoints/best_val_00000026_000000002646.ckpt"
)


def load_model(ckpt_path: str, device: torch.device) -> Proteina:
    logger.info(f"Loading {ckpt_path}")
    model = Proteina.load_from_checkpoint(
        ckpt_path, strict=False, autoencoder_ckpt_path=None
    )
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if hasattr(model.nn, "_orig_mod"):
        model.nn = model.nn._orig_mod
        logger.info("  unwrapped torch.compile wrapper")
    return model


def sample_one(model: Proteina, L_target: int, seed: int, nsteps: int,
               device: torch.device) -> torch.Tensor:
    """Sample one protein at length L_target with the given seed.
    Returns final bb_ca coords [n_pad, 3] in nm."""
    torch.manual_seed(seed)
    n_pad = max(L_target + 16, 64)
    # B=2 pad: full_simulation has a squeeze+assert path that fails on B=1
    # (product_space_flow_matcher.py:626-632) — same pad-trick as the
    # dense_self_sparsify and hybrid_grad scripts.
    B_use = 2
    mask = torch.zeros(1, n_pad, dtype=torch.bool, device=device)
    mask[0, :L_target] = True
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
    fn_predict_for_sampling = _partial(model.predict_for_sampling, n_recycle=0)
    sc_neighbors_active = model.cfg_exp.training.get("sc_neighbors", False)

    with torch.no_grad():
        gen_samples, _info = model.fm.full_simulation(
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
    return final[0]  # [n_pad, 3]


def save_pdb_ca_only(coords_ca: torch.Tensor, L_target: int, path: str):
    """Write CA-only PDB (UNK residues, chain A). Matches the format used by
    dense_self_sparsify_inference.py so the eval pipeline is uniform."""
    coords_ang = coords_ca[:L_target].detach().cpu().numpy() * 10.0
    lines = []
    atom_serial = 1
    chain = "A"
    for i in range(L_target):
        x, y, z = coords_ang[i]
        lines.append(
            f"ATOM  {atom_serial:>5d}  CA  ALA {chain}{i+1:>4d}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C"
        )
        atom_serial += 1
    lines.append("END")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--sparse_ckpt", required=True,
                    help="Trained Move-2 router-sparse trunk checkpoint.")
    ap.add_argument("--dense_ckpt", default=DENSE_CKPT_DEFAULT,
                    help="Canonical dense baseline (E019). Default is the on-disk best_val ckpt.")
    ap.add_argument("--n_samples", type=int, default=12,
                    help="Samples per length. Default 12 for full eval; use 6 for probes.")
    ap.add_argument("--lengths", default="100,200",
                    help="Comma-sep lengths. Default {100, 200} per prompt.")
    ap.add_argument("--nsteps", type=int, default=400,
                    help="ODE steps. nsteps=400 is the CLAUDE.md hard rule.")
    ap.add_argument("--seed", type=int, default=5,
                    help="Base seed; per-(sample_idx, L) seed = base + 1000*idx + L.")
    ap.add_argument("--out_dir", default="results/sparse_trunk_move2",
                    help="Parent dir; PDBs land at <out_dir>/<label>/{sparse,baseline}/L*/")
    ap.add_argument("--baseline_only", action="store_true",
                    help="Skip sparse arm; only sample baseline (useful if baseline ckpt missing).")
    ap.add_argument("--sparse_only", action="store_true",
                    help="Skip baseline arm; only sample sparse.")
    args = ap.parse_args()

    load_dotenv()
    L.seed_everything(args.seed)
    assert args.nsteps == 400, "nsteps must be 400 per CLAUDE.md hard rule."

    lengths = [int(x) for x in args.lengths.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Sampling requires CUDA."

    out_root = Path(args.out_dir) / args.label
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 80)
    logger.info(f"Move-2 paired sampling | label={args.label}")
    logger.info(f"  sparse_ckpt: {args.sparse_ckpt}")
    logger.info(f"  dense_ckpt:  {args.dense_ckpt}")
    logger.info(f"  N={args.n_samples} × L={lengths} × nsteps={args.nsteps}")
    logger.info(f"  out_root:    {out_root}")
    logger.info("=" * 80)

    sparse_model = None if args.baseline_only else load_model(args.sparse_ckpt, device)
    dense_model = None if args.sparse_only else load_model(args.dense_ckpt, device)
    if sparse_model is not None:
        # Sanity-check the loaded model has router_sparse_K set.
        nn_cfg = sparse_model.cfg_exp.get("nn", {})
        assert nn_cfg.get("router_sparse_K", None) is not None, (
            f"--sparse_ckpt {args.sparse_ckpt} has no router_sparse_K in its config — "
            f"likely the wrong ckpt (canonical dense?). Use --baseline_only if intended."
        )
        logger.info(f"  sparse trunk: router_sparse_K={nn_cfg.get('router_sparse_K')}")

    metadata = []
    for L_target in lengths:
        for prot_idx in range(args.n_samples):
            seed = args.seed + 1000 * prot_idx + L_target
            logger.info(f"--- L={L_target} sample {prot_idx} (seed={seed}) ---")

            if sparse_model is not None:
                t0 = time.time()
                coords = sample_one(sparse_model, L_target, seed, args.nsteps, device)
                wall = time.time() - t0
                p = out_root / "sparse" / f"L{L_target}" / f"sample_{prot_idx}.pdb"
                p.parent.mkdir(parents=True, exist_ok=True)
                save_pdb_ca_only(coords, L_target, str(p))
                logger.info(f"  sparse   saved {p} ({wall:.1f}s)")
                metadata.append({"arm": "sparse", "L": L_target, "sample_idx": prot_idx,
                                 "seed": seed, "path": str(p), "wall_s": wall})

            if dense_model is not None:
                t0 = time.time()
                coords = sample_one(dense_model, L_target, seed, args.nsteps, device)
                wall = time.time() - t0
                p = out_root / "baseline" / f"L{L_target}" / f"sample_{prot_idx}.pdb"
                p.parent.mkdir(parents=True, exist_ok=True)
                save_pdb_ca_only(coords, L_target, str(p))
                logger.info(f"  baseline saved {p} ({wall:.1f}s)")
                metadata.append({"arm": "baseline", "L": L_target, "sample_idx": prot_idx,
                                 "seed": seed, "path": str(p), "wall_s": wall})

    meta_path = out_root / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "label": args.label,
            "sparse_ckpt": args.sparse_ckpt,
            "dense_ckpt": args.dense_ckpt,
            "n_samples": args.n_samples,
            "lengths": lengths,
            "nsteps": args.nsteps,
            "seed": args.seed,
            "samples": metadata,
        }, f, indent=2)
    logger.info(f"Wrote {meta_path}")
    logger.info("Done. Next: script_utils/eval_router_sparse_paired.py --label " + args.label)


if __name__ == "__main__":
    main()
