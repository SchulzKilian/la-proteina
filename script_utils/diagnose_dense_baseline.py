"""Diagnostic: call dense's full_simulation with the canonical seed scheme
and a single-batch N=6 setup (no B=1+pad workaround, no per-sample re-seed).

If this script's output ≈ canonical Hydra (`generate.py
inference_canonical_step2646_n6_nfe400`), then the bug in
`dense_self_sparsify_per_lh_inference.py`'s baseline is the per-sample
re-seed + B=1→B=2 pad combination.

If this script's output is still degraded vs canonical, the bug is somewhere
else (sampling_model_args dict, predict_for_sampling shape, batch keys).

Usage:
    CUDA_VISIBLE_DEVICES=1 python script_utils/diagnose_dense_baseline.py
"""
import os
import sys
import time
from pathlib import Path

REPO = "/home/ks2218/la-proteina"
sys.path.insert(0, REPO)

import lightning as L
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from functools import partial

from proteinfoundation.proteina import Proteina


DENSE_CKPT = "/home/ks2218/la-proteina/best_val_00000026_000000002646.ckpt"


def main():
    load_dotenv()
    device = torch.device("cuda")

    # Match canonical inference_base.yaml verbatim.
    sampling_model_args = OmegaConf.create({
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

    logger.info(f"Loading dense ckpt: {DENSE_CKPT}")
    model = Proteina.load_from_checkpoint(
        DENSE_CKPT, strict=False, autoencoder_ckpt_path=None
    )
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # IMPORTANT: do NOT unwrap torch.compile. Canonical inference keeps it.
    # (The script_utils/dense_self_sparsify_per_lh_inference.py unwraps because
    # the patched _build_neighbor_idx can't survive dynamo; for the baseline-
    # only diagnostic we keep it.)
    logger.info(
        f"  model.nn type: {type(model.nn).__name__}  "
        f"(compiled={hasattr(model.nn, '_orig_mod')})"
    )

    out_root = Path("results/diagnose_dense_baseline/canonical_seed_n6_L50100")
    out_root.mkdir(parents=True, exist_ok=True)

    fn_predict_for_sampling = partial(model.predict_for_sampling, n_recycle=0)
    sc_active = model.cfg_exp.training.get("sc_neighbors", False)

    for batch_idx, L_target in enumerate([50, 100]):
        seed = 5 + batch_idx  # canonical: _generation_base_seed=5 + batch_idx
        L.seed_everything(seed)

        nsamples = 6
        n_pad = max(L_target + 16, 64)
        mask = torch.zeros(nsamples, n_pad, dtype=torch.bool, device=device)
        mask[:, :L_target] = True

        batch = {"nsamples": nsamples, "nres": n_pad, "mask": mask}

        logger.info(
            f"=== L={L_target} batch_idx={batch_idx} seed={seed} "
            f"nsamples={nsamples} n_pad={n_pad} ==="
        )
        t0 = time.time()
        with torch.no_grad():
            gen_samples, extra_info = model.fm.full_simulation(
                batch=batch,
                predict_for_sampling=fn_predict_for_sampling,
                nsteps=400,
                nsamples=nsamples,
                n=n_pad,
                self_cond=True,
                sampling_model_args=sampling_model_args,
                device=device,
                save_trajectory_every=0,
                guidance_w=1.0,
                ag_ratio=0.0,
                steering_guide=None,
                sc_neighbors_active=sc_active,
                sc_neighbors_bootstrap=True,
            )
        wall = time.time() - t0
        logger.info(f"  gen wall: {wall:.1f}s")
        # gen_samples["bb_ca"] expected [nsamples, n_pad, 3]
        bb_ca = gen_samples["bb_ca"] if isinstance(gen_samples, dict) else gen_samples
        if isinstance(bb_ca, dict):
            bb_ca = bb_ca.get("bb_ca", next(iter(bb_ca.values())))
        # Save PDBs.
        from script_utils.dense_self_sparsify_per_lh_inference import save_pdb_ca_only
        bdir = out_root / f"L{L_target}"
        bdir.mkdir(parents=True, exist_ok=True)
        rtype = torch.zeros(L_target, dtype=torch.long)
        for s in range(nsamples):
            coords = bb_ca[s, :L_target]
            m = mask[s, :L_target].cpu()
            save_pdb_ca_only(coords, m, rtype, str(bdir / f"sample_{s}.pdb"))
        logger.info(f"  saved 6 PDBs to {bdir}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
