"""Hybrid generation: switch between two CA-only Proteina checkpoints at t=t_switch.

Convention follows the codebase: t=0 is noise, t=1 is clean. With t_switch=0.75:
  - model_A runs for steps where t < 0.75 (bulk denoising)
  - model_B runs for steps where t >= 0.75 (final detailing)

Both checkpoints must be CA-only (`local_latents` absent from product_flowmatcher).
The integrator is taken from model_B; predict_for_sampling is monkey-patched on
model_B to dispatch to model_A early.
"""
import os
import sys
import types
from typing import Dict

root = os.path.abspath(".")
sys.path.insert(0, root)
# isort: split

import hydra
import lightning as L
import torch
from omegaconf import OmegaConf
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader

from proteinfoundation.datasets.gen_dataset import GenDataset
from proteinfoundation.proteina import Proteina
from proteinfoundation.generate import (
    setup,
    save_predictions,
    check_cfg_validity,
)

torch.set_float32_matmul_precision("high")


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="inference_hybrid_conv_to_scnbr_quick",
)
def main(cfg: Dict) -> None:
    load_dotenv()

    config_name = cfg.get("config_name", "inference_hybrid_conv_to_scnbr_quick")
    job_id = cfg.get("job_id", 0)

    root_path = setup(cfg, create_root=True, config_name=config_name, job_id=job_id)
    OmegaConf.save(cfg, os.path.join(root_path, "resolved_config.yaml"))

    csv_filename = f"results_{config_name}_{job_id}.csv"
    csv_path = os.path.join(root_path, "..", csv_filename)
    if os.path.exists(csv_path):
        logger.info(f"Results already exist at {csv_path}. Exiting.")
        sys.exit(0)

    cfg_gen = cfg.generation
    check_cfg_validity(cfg_gen.dataset, cfg_gen.args)

    t_switch = float(cfg.t_switch)
    ckpt_a = os.path.join(cfg.ckpt_path_a, cfg.ckpt_name_a)
    ckpt_b = os.path.join(cfg.ckpt_path_b, cfg.ckpt_name_b)
    assert os.path.exists(ckpt_a), f"missing ckpt A: {ckpt_a}"
    assert os.path.exists(ckpt_b), f"missing ckpt B: {ckpt_b}"
    logger.info(
        f"Hybrid sampling | A (t<{t_switch}) = {ckpt_a} | B (t>={t_switch}) = {ckpt_b}"
    )

    model_A = Proteina.load_from_checkpoint(
        ckpt_a, strict=False, autoencoder_ckpt_path=None
    )
    model_B = Proteina.load_from_checkpoint(
        ckpt_b, strict=False, autoencoder_ckpt_path=None
    )

    run_a = model_A.cfg_exp.get("run_name_")
    run_b = model_B.cfg_exp.get("run_name_")
    logger.info(f"  model_A.run_name_ = {run_a}")
    logger.info(f"  model_B.run_name_ = {run_b}")
    assert "local_latents" not in model_A.cfg_exp.product_flowmatcher, (
        "model_A is not CA-only"
    )
    assert "local_latents" not in model_B.cfg_exp.product_flowmatcher, (
        "model_B is not CA-only"
    )

    # configure_inference for both (sets self.inf_cfg used by predict_step on B,
    # and would be used by A if anything inside it reads inf_cfg).
    model_A.configure_inference(cfg_gen, nn_ag=None)
    model_B.configure_inference(cfg_gen, nn_ag=None)

    # Place model_A on the same GPU model_B will land on. Lightning will move B.
    device = torch.device("cuda")
    model_A.to(device).eval()

    # Dispatch counters and per-batch kink stats. We log a kink each batch.
    counter = {"A": 0, "B": 0, "first_switch_logged_this_batch": False}
    kink_log = []  # list of dicts, one per batch; flushed/echoed after run

    def reset_per_batch_state():
        counter["first_switch_logged_this_batch"] = False

    def compute_kink(batch, t_val):
        """At the handover step: compute v_A and v_B on the SAME (x_t, t),
        return per-protein ‖v_A − v_B‖ stats + norms + cosine.

        Both models output bb_ca with output_parameterization='v', so the raw
        nn_out["bb_ca"]["v"] tensor is the velocity. Mask is applied so padded
        residues don't contribute."""
        with torch.no_grad():
            nn_out_A = model_A.predict_for_sampling(batch, mode="full", n_recycle=0)
            nn_out_B = Proteina.predict_for_sampling(
                model_B, batch, mode="full", n_recycle=0
            )
        v_A = nn_out_A["bb_ca"]["v"].float()  # [B, N, 3]
        v_B = nn_out_B["bb_ca"]["v"].float()
        mask = batch["mask"].float().unsqueeze(-1)  # [B, N, 1]

        diff = (v_A - v_B) * mask
        per_prot_diff = diff.flatten(1).norm(dim=-1)  # [B]
        per_prot_a = (v_A * mask).flatten(1).norm(dim=-1)
        per_prot_b = (v_B * mask).flatten(1).norm(dim=-1)
        # Cosine — flatten valid residues only, but using masked vector is fine
        # because padded slots are zeroed in both numerator and denominator.
        a_flat = (v_A * mask).flatten(1)
        b_flat = (v_B * mask).flatten(1)
        cos = (a_flat * b_flat).sum(-1) / (
            a_flat.norm(dim=-1).clamp(min=1e-12) * b_flat.norm(dim=-1).clamp(min=1e-12)
        )
        # Per-residue: average ‖v_A_i − v_B_i‖ over valid residues
        per_res_diff = (v_A - v_B).norm(dim=-1)  # [B, N]
        nres = mask.squeeze(-1).sum(-1).clamp(min=1.0)  # [B]
        per_res_diff_mean = (per_res_diff * mask.squeeze(-1)).sum(-1) / nres

        n_proteins = v_A.shape[0]
        record = {
            "t_handover": float(t_val),
            "n_proteins": int(n_proteins),
            "diff_l2_per_protein": [float(x) for x in per_prot_diff.tolist()],
            "vA_l2_per_protein": [float(x) for x in per_prot_a.tolist()],
            "vB_l2_per_protein": [float(x) for x in per_prot_b.tolist()],
            "cos_per_protein": [float(x) for x in cos.tolist()],
            "diff_per_residue_mean_per_protein": [
                float(x) for x in per_res_diff_mean.tolist()
            ],
        }
        # Aggregate summary
        record["mean_diff_l2"] = float(per_prot_diff.mean().item())
        record["mean_vA_l2"] = float(per_prot_a.mean().item())
        record["mean_vB_l2"] = float(per_prot_b.mean().item())
        record["mean_cos"] = float(cos.mean().item())
        record["mean_diff_per_residue"] = float(per_res_diff_mean.mean().item())
        record["mean_relative_diff"] = float(
            (per_prot_diff / per_prot_a.clamp(min=1e-12)).mean().item()
        )
        kink_log.append(record)
        logger.info(
            f"[hybrid kink] t_handover={t_val:.4f} "
            f"‖v_A−v_B‖_2={record['mean_diff_l2']:.3f} "
            f"‖v_A‖={record['mean_vA_l2']:.3f} ‖v_B‖={record['mean_vB_l2']:.3f} "
            f"cos(v_A,v_B)={record['mean_cos']:.4f} "
            f"per-res‖Δv‖={record['mean_diff_per_residue']:.3f} "
            f"rel={record['mean_relative_diff']:.3f}"
        )

    def hybrid_predict_for_sampling(self, batch, mode, n_recycle):
        t_val = batch["t"]["bb_ca"].flatten()[0].item()
        if t_val < t_switch:
            counter["A"] += 1
            return model_A.predict_for_sampling(batch, mode, n_recycle)
        else:
            counter["B"] += 1
            if not counter["first_switch_logged_this_batch"]:
                logger.info(
                    f"[hybrid canary] first scnbr (B) call at t={t_val:.4f} "
                    f"(after {counter['A']} conv (A) calls so far)"
                )
                # Log the kink BEFORE returning B's output. The two extra
                # forwards are fine — outer loop is in torch.no_grad and we
                # only do this once per batch.
                # Note: only log for mode=='full'. With CFG (guidance_w!=1.0)
                # the integrator would also call mode=='ucond' at the same
                # step; we skip those to avoid double logging.
                if mode == "full":
                    compute_kink(batch, t_val)
                counter["first_switch_logged_this_batch"] = True
            return Proteina.predict_for_sampling(self, batch, mode, n_recycle)

    model_B.predict_for_sampling = types.MethodType(
        hybrid_predict_for_sampling, model_B
    )

    # Wrap predict_step so we reset the per-batch kink-log flag at each batch.
    _orig_predict_step = model_B.predict_step

    def hybrid_predict_step(self, batch, batch_idx):
        reset_per_batch_state()
        return _orig_predict_step(batch, batch_idx)

    model_B.predict_step = types.MethodType(hybrid_predict_step, model_B)

    model_B._generation_base_seed = cfg.seed

    motif_cond = cfg_gen.args.get("motif_cond", False)
    if motif_cond or "motif_task_name" in cfg_gen.dataset:
        raise NotImplementedError(
            "Hybrid sampling does not currently support motif/conditional generation"
        )
    dataset = GenDataset(**cfg_gen.dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    trainer = L.Trainer(accelerator="gpu", devices=1)
    predictions = trainer.predict(model_B, dataloader)

    # Persist the kink log next to the run output for downstream analysis.
    import json
    kink_path = os.path.join(root_path, "kink_log.json")
    with open(kink_path, "w") as f:
        json.dump(kink_log, f, indent=2)
    logger.info(f"[hybrid kink] wrote {len(kink_log)} batch records to {kink_path}")

    logger.info(
        f"[hybrid summary] dispatch counts: A (conv) = {counter['A']}, "
        f"B (scnbr) = {counter['B']}"
    )

    save_predictions(
        root_path,
        predictions,
        job_id=job_id,
        chain_indexes=None,
        cath_codes=dataset.cath_codes,
    )


if __name__ == "__main__":
    main()
