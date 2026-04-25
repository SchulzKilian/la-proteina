"""Generate guided + unguided proteins for steering evaluation.

Usage:
    python -m steering.generate \
        --proteina_config inference_ucond_notri_long \
        --steering_config steering/config/examples/net_charge_up.yaml \
        --n_samples 5 --lengths 300 400 --seeds 42 43 44 45 46 \
        --output_dir results/steering_eval/net_charge_up \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure project root on path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import lightning as L
from omegaconf import OmegaConf

from proteinfoundation.generate import load_ckpt_n_configure_inference, parse_args_and_cfg
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb
from steering.guide import SteeringGuide

logger = logging.getLogger(__name__)

# OpenFold restypes for sequence conversion
from openfold.np.residue_constants import restypes, restype_1to3

IDX_TO_AA = {i: aa for i, aa in enumerate(restypes)}
IDX_TO_AA[20] = "X"


def load_proteina_config(config_name: str) -> dict:
    """Load a Hydra config by name, using OmegaConf compose."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    config_dir = str(_ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


def generate_one(
    model,
    length: int,
    seed: int,
    device: torch.device,
) -> tuple:
    """Generate a single protein of given length with given seed.

    Returns:
        (coors_atom37 [n, 37, 3], residue_type [n], extra_info dict)
    """
    L.seed_everything(seed)

    # Build a minimal batch for generation
    batch = {
        "nsamples": 1,
        "nres": length,
    }

    # Set the base seed so predict_step re-seeds deterministically
    model._generation_base_seed = seed

    # Call predict_step directly (bypasses Lightning trainer overhead)
    model.eval()
    with torch.no_grad():
        from functools import partial

        self_cond = model.inf_cfg.args.self_cond
        nsteps = model.inf_cfg.args.nsteps
        guidance_w = model.inf_cfg.args.get("guidance_w", 1.0)
        ag_ratio = model.inf_cfg.args.get("ag_ratio", 0.0)

        fn_predict = partial(
            model.predict_for_sampling,
            n_recycle=model.inf_cfg.get("n_recycle", 0),
        )

        gen_samples, extra_info = model.fm.full_simulation(
            batch=batch,
            predict_for_sampling=fn_predict,
            nsteps=nsteps,
            nsamples=1,
            n=length,
            self_cond=self_cond,
            sampling_model_args=model.inf_cfg.model,
            device=device,
            guidance_w=guidance_w,
            ag_ratio=ag_ratio,
            steering_guide=getattr(model, "steering_guide", None),
        )

        sample_prots = model.sample_formatting(
            x=gen_samples, extra_info=extra_info, ret_mode="coors37_n_aatype",
        )

    coors = sample_prots["coors"][0]        # [n, 37, 3] Angstroms
    res_type = sample_prots["residue_type"][0]  # [n]
    mask = sample_prots["mask"][0]              # [n]

    return coors, res_type, mask, extra_info


def save_protein(
    coors: torch.Tensor,
    residue_type: torch.Tensor,
    mask: torch.Tensor,
    protein_id: str,
    out_dir: Path,
):
    """Save protein as both PDB and .pt (for property computation)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # PDB
    pdb_path = out_dir / f"{protein_id}.pdb"
    write_prot_to_pdb(
        prot_pos=coors.float().detach().cpu().numpy(),
        aatype=residue_type.detach().cpu().numpy(),
        file_path=str(pdb_path),
        overwrite=True,
        no_indexing=True,
    )

    # .pt — save in a format that property_evaluate.py can use directly.
    # Coordinates are already in OpenFold atom order from the decoder.
    # We save as a plain dict (not PyG Data) with explicit field names
    # that won't trigger the PDB->OpenFold reindex in compute_developability.
    L = int(mask.sum().item())
    coors_cpu = coors[:L].detach().cpu()
    res_type_cpu = residue_type[:L].detach().cpu()

    # Build atom mask: atoms with non-zero coords are resolved
    coord_mask = (coors_cpu.abs().sum(dim=-1) > 1e-6)  # [L, 37]

    # Build residue names
    seq = "".join(IDX_TO_AA.get(int(idx), "X") for idx in res_type_cpu)
    residue_names = [restype_1to3.get(aa, "UNK") for aa in seq]

    pt_path = out_dir / f"{protein_id}.pt"
    torch.save({
        "coords_openfold": coors_cpu,        # [L, 37, 3] already OpenFold order
        "coord_mask": coord_mask,             # [L, 37]
        "residue_type": res_type_cpu,         # [L]
        "residues": residue_names,            # list of 3-letter strings
        "id": protein_id,
        "sequence": seq,
    }, pt_path)

    return pdb_path, pt_path


def main():
    parser = argparse.ArgumentParser(description="Generate guided + unguided proteins")
    parser.add_argument("--proteina_config", type=str, required=True,
                        help="Name of La-Proteina inference config (e.g. inference_ucond_notri_long)")
    parser.add_argument("--steering_config", type=str, required=True,
                        help="Path to steering YAML config file")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of seeds (proteins per length)")
    parser.add_argument("--lengths", type=int, nargs="+", default=[300, 400],
                        help="Protein lengths to generate")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Explicit seeds (overrides --n_samples)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--nsteps", type=int, default=None,
                        help="Override number of ODE steps (default: from config)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Seeds
    if args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = list(range(42, 42 + args.n_samples))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load La-Proteina model
    logger.info("Loading La-Proteina config: %s", args.proteina_config)
    cfg = load_proteina_config(args.proteina_config)

    if args.nsteps is not None:
        cfg.generation.args.nsteps = args.nsteps

    logger.info("Loading model checkpoint...")
    model = load_ckpt_n_configure_inference(cfg)
    model = model.to(device)
    model.eval()

    # Load steering config
    with open(args.steering_config) as f:
        steering_cfg = yaml.safe_load(f)["steering"]

    # Set device for steering predictor to match model
    steering_cfg["device"] = str(device)

    # Build steering guide
    guide = SteeringGuide(steering_cfg)
    logger.info("Steering: enabled=%s, objectives=%s", guide.enabled, steering_cfg.get("objectives", []))

    # Create output directories
    output_dir = Path(args.output_dir)
    guided_dir = output_dir / "guided"
    unguided_dir = output_dir / "unguided"
    diag_dir = output_dir / "diagnostics"
    guided_dir.mkdir(parents=True, exist_ok=True)
    unguided_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    run_config = {
        "proteina_config": args.proteina_config,
        "steering_config": args.steering_config,
        "steering": steering_cfg,
        "lengths": args.lengths,
        "seeds": seeds,
        "nsteps": cfg.generation.args.nsteps,
        "device": str(device),
    }
    with open(output_dir / "run_config.yaml", "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)

    # Generate
    total = len(seeds) * len(args.lengths)
    done = 0
    t0 = time.time()

    for length in args.lengths:
        for seed in seeds:
            protein_id = f"s{seed}_n{length}"
            done += 1
            logger.info("[%d/%d] Generating %s ...", done, total, protein_id)

            # --- Unguided ---
            model.steering_guide = None
            coors_u, res_u, mask_u, _ = generate_one(model, length, seed, device)
            save_protein(coors_u, res_u, mask_u, protein_id, unguided_dir)

            # --- Guided ---
            model.steering_guide = guide
            coors_g, res_g, mask_g, extra_g = generate_one(model, length, seed, device)
            save_protein(coors_g, res_g, mask_g, protein_id, guided_dir)

            # Save diagnostics
            diag = extra_g.get("steering_diagnostics", [])
            diag_path = diag_dir / f"{protein_id}_diagnostics.json"
            # Convert to serialisable format
            diag_serialisable = []
            for d in diag:
                entry = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        entry[k] = {kk: float(vv) for kk, vv in v.items()}
                    elif isinstance(v, (int, float, bool, str)):
                        entry[k] = v
                    else:
                        entry[k] = str(v)
                diag_serialisable.append(entry)
            with open(diag_path, "w") as f:
                json.dump(diag_serialisable, f, indent=2)

            elapsed = time.time() - t0
            rate = done / elapsed
            remaining = (total - done) / rate if rate > 0 else 0
            logger.info("  Done. %.1fs elapsed, ~%.0fs remaining", elapsed, remaining)

    logger.info("Generation complete. Output: %s", output_dir)
    logger.info("  Guided:    %s", guided_dir)
    logger.info("  Unguided:  %s", unguided_dir)
    logger.info("  Diagnostics: %s", diag_dir)


if __name__ == "__main__":
    main()
