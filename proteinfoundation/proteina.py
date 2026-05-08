import os
import random
from functools import partial
from typing import Dict, List, Literal, Tuple, Union

import lightning as L
import numpy as np
import torch
from jaxtyping import Bool, Float
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger
from torch import Tensor
from omegaconf import OmegaConf

from proteinfoundation.flow_matching.product_space_flow_matcher import (
    ProductSpaceFlowMatcher,
)
from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer
from proteinfoundation.nn.local_latents_transformer_unindexed import LocalLatentsTransformerMotifUidx
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.utils.coors_utils import nm_to_ang, trans_nm_to_atom37, ca_nm_to_backbone_atom37
from proteinfoundation.utils.pdb_utils import (
    create_full_prot,
    to_pdb,
)


@rank_zero_only
def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


class Proteina(L.LightningModule):
    def __init__(self, cfg_exp, store_dir=None, autoencoder_ckpt_path=None, use_precomputed_latents=False):

        super().__init__()
        self.save_hyperparameters()
        self.cfg_exp = cfg_exp
        self.inf_cfg = None  # Only for inference runs
        self.validation_output_lens = {}
        self.validation_output_data = []
        self.store_dir = store_dir if store_dir is not None else "./tmp"
        self.val_path_tmp = os.path.join(self.store_dir, "val_samples")
        create_dir(self.val_path_tmp)

        self.metric_factory = None
        self.use_precomputed_latents = cfg_exp.training.get("use_precomputed_latents", False)

        if autoencoder_ckpt_path is not None:
            # Allow adding new keys
            logger.info(f"Manually setting autoencoder_ckpt_path to {autoencoder_ckpt_path}")
            OmegaConf.set_struct(cfg_exp, False)
            # Update the configuration with the new key-value pair
            cfg_exp.autoencoder_ckpt_path = autoencoder_ckpt_path
            # Re-enable struct mode if needed
            OmegaConf.set_struct(cfg_exp, True)

        DEBUG_AE = False

        self._ca_only_mode = "local_latents" not in cfg_exp.product_flowmatcher

        if self._ca_only_mode:
            # CA-only mode: no autoencoder, no latents
            logger.info("CA-only mode detected (no 'local_latents' in product_flowmatcher). Skipping AutoEncoder.")
            # Use .get() — CA-only training configs may not have autoencoder_ckpt_path in struct at all
            if cfg_exp.get("autoencoder_ckpt_path", None) is not None:
                OmegaConf.set_struct(cfg_exp, False)
                cfg_exp.autoencoder_ckpt_path = None
                OmegaConf.set_struct(cfg_exp, True)
            self.autoencoder = None
            self.latent_dim = None
        elif self.use_precomputed_latents and not DEBUG_AE:
            assert "motif" not in cfg_exp.nn.name.lower(), \
                f"FATAL: Motif scaffolding (NN: {cfg_exp.nn.name}) requires full 37-atom coordinates, but precomputed latents discard them. Disable precomputed latents."
            logger.info("Skipping AutoEncoder load -> using precomputed latents.")
            self.autoencoder = None
            latent_dim = cfg_exp.product_flowmatcher.local_latents.get("dim", 8)
            # OmegaConf returns None when key exists with `null` value (not the default).
            # Fall back to 8 (standard AE latent dim) so downstream modules get a valid int.
            if latent_dim is None:
                latent_dim = 8
                logger.warning(
                    "product_flowmatcher.local_latents.dim is null in config; "
                    "defaulting to latent_dim=8 for precomputed latents. "
                    "Set it explicitly in the config to suppress this warning."
                )
            self.latent_dim = latent_dim
            # Write resolved dim back so the flow matcher and NN are instantiated correctly,
            # mirroring what the normal AE path does.
            try:
                cfg_exp.product_flowmatcher.local_latents.dim = self.latent_dim
            except:
                OmegaConf.set_struct(cfg_exp, False)
                cfg_exp.product_flowmatcher.local_latents.dim = self.latent_dim
                OmegaConf.set_struct(cfg_exp, True)
        else:
            # Original AutoEncoder loading
            self.autoencoder, latent_dim = self.load_autoencoder(cfg_exp, freeze_params=True)

            # Add right latent dimensionality in the config file, needed to instantiate the flow matcher below
            if latent_dim is not None:
                self.latent_dim = latent_dim
            else:
                self.latent_dim = cfg_exp.product_flowmatcher.local_latents.get("dim", 8)

            if self.autoencoder is not None:
                try:
                    cfg_exp.product_flowmatcher.local_latents.dim = self.latent_dim
                except:
                    OmegaConf.set_struct(cfg_exp, False)
                    cfg_exp.product_flowmatcher.local_latents.dim = self.latent_dim
                    OmegaConf.set_struct(cfg_exp, True)

        self.fm = ProductSpaceFlowMatcher(cfg_exp)
        logger.info(f"cfg_exp.nn: {cfg_exp.nn}")

        # Neural network
        # Fix C2: source of truth for sc_neighbors lives under cfg_exp.training (alongside
        # self_cond), but the NN constructor only sees cfg_exp.nn. Plumb the resolved
        # values through as kwargs — defaults preserve existing behavior when absent.
        # Same pattern for curriculum_neighbors (K=64 t-bucketed neighbor reallocation):
        # source of truth is cfg_exp.training, plumbed as a kwarg so existing configs
        # without the flag stay unchanged.
        _sc_neighbors_kwargs = {
            "sc_neighbors": cfg_exp.training.get("sc_neighbors", False),
            "sc_neighbors_t_threshold": cfg_exp.training.get("sc_neighbors_t_threshold", 0.4),
            "curriculum_neighbors": cfg_exp.training.get("curriculum_neighbors", False),
        }
        if cfg_exp.nn.name == "local_latents_transformer":
            self.nn = LocalLatentsTransformer(
                **cfg_exp.nn, latent_dim=self.latent_dim, **_sc_neighbors_kwargs
            )
        elif cfg_exp.nn.name == "local_latents_transformer_motif_uidx":
            self.nn = LocalLatentsTransformerMotifUidx(**cfg_exp.nn, latent_dim=self.latent_dim)
        else:
            raise IOError(f"Wrong nn selected for CAFlow {cfg_exp.nn.name}")

        # Optional torch.compile of the trunk. fullgraph=False so the per-bucket
        # Python loop in _build_neighbor_idx (curriculum path) graph-breaks
        # cleanly without erroring; mode="reduce-overhead" is the steady-state
        # training mode (one-time compile, reuse compiled graph). Defaults off
        # so existing checkpoints / runs are unchanged.
        if cfg_exp.opt.get("compile_nn", False):
            logger.info("opt.compile_nn=True — wrapping self.nn with torch.compile (mode=reduce-overhead, fullgraph=False).")
            self.nn = torch.compile(self.nn, mode="reduce-overhead", fullgraph=False)

        # Scaling laws stuff
        self.nflops = 0
        self.nsamples_processed = 0
        self.nparams = sum(p.numel() for p in self.nn.parameters() if p.requires_grad)

        self.nn_ag = None

        # Inside Proteina or AutoEncoder class
    def verify_latent_consistency(self, batch, on_the_fly_mean):
        """
        Handles shape mismatches between OTF calculations (usually [B, L, 8])
        and Disk-saved precomputed latents (usually [L, 8]).
        """
        if self.global_step % 1 != 0:
            pass
            # return

        # 1. Correct Access to Batch Info
        # Use dictionary syntax [] instead of . notation to avoid AttributeErrors
        try:
            protein_id = batch["id"][0] if "id" in batch else batch["pdb"][0]
        except (KeyError, TypeError):
            return

        shard = protein_id[0:2].lower()

        base_dir = "/rds/user/ks2218/hpc-work/processed_latents"
        latent_path = os.path.join(base_dir, shard, f"{protein_id}.pt")
        # Note: Use the actual data directory from your config
        # latent_path = os.path.join("data/pdb_train/processed_latents", f"{protein_id}.pt")

        if not os.path.exists(latent_path):

            return

        # 2. Load and Resolve Disk Shape
        precomputed_data = torch.load(latent_path, map_location=self.device, weights_only=False)
        disk_mean = precomputed_data.mean.to(self.device) # Expected [L, 8]
        
        # 3. Resolve OTF Shape
        # on_the_fly_mean is passed as batch["x_1"]["local_latents"], typically [B, L, 8]
        # print(on_the_fly_mean.shape)
        otf_sample_mean = on_the_fly_mean[0] # Take first sample in batch -> [L, 8]

        # 4. Critical Shape Alignment (The "Shape Error" Fix)
        # Ensure both are [L, 8]. Sometimes precompute scripts save [8, L] by mistake.
        if disk_mean.shape[1] != 8 and disk_mean.shape[0] == 8:
            disk_mean = disk_mean.transpose(0, 1)
            
        # 5. Length Alignment
        # The batch might have padding. Use the mask to get the real sequence length.
        mask = batch["mask"][0] # [L] boolean mask
        
        # Mask both tensors to compare only the valid protein residues
        otf_valid = otf_sample_mean[mask]
        disk_valid = disk_mean[mask]

        # Final check before subtraction
        if otf_valid.shape != disk_valid.shape:
            print(f"❌ Shape Mismatch for {protein_id}: OTF {otf_valid.shape} vs Disk {disk_valid.shape}")
            return

        diff = torch.abs(otf_valid - disk_valid).max().item()
        
        if diff > 1e-4:
            print(f"⚠️  LATENT MISMATCH at step {self.global_step} for {protein_id}! Max Diff: {diff:.6f}")
        else:
            print(f"✅ Latent Match for {protein_id} (Diff: {diff:.2e})")

        raise Exception()


    def load_autoencoder(self, cfg_exp, freeze_params=True):
        """Loads autoencoder, if required."""
        if ("autoencoder_ckpt_path" in cfg_exp):
            # for new runs trained with refactored codebase
            ae_ckp_path = cfg_exp.autoencoder_ckpt_path
        elif ("autoencoder_ckpt_path" in cfg_exp.product_flowmatcher.local_latents):
            # for old runs trained with old codebase
            ae_ckp_path = cfg_exp.product_flowmatcher.local_latents.autoencoder_ckpt_path
        else:
            raise ValueError("No autoencoder checkpoint path provided")

        if ae_ckp_path is None:
            return None, None
        

        logger.info(f"Loading autoencoder from {ae_ckp_path}")
        autoencoder = AutoEncoder.load_from_checkpoint(ae_ckp_path, strict=False)
        if freeze_params:
            for param in autoencoder.parameters():
                param.requires_grad = False
        return autoencoder, autoencoder.latent_dim

    def configure_optimizers(self):
        opt_cfg = self.cfg_exp.opt
        weight_decay = float(opt_cfg.get("weight_decay", 0.0) or 0.0)
        # Standard DiT/SiT/SD3 pattern: split into wd and no-wd parameter groups.
        # The no-wd group covers parameters that either (a) have no overfitting
        # capacity (biases, LayerNorm γ/β, embeddings) or (b) initialize at zero
        # and need to grow against the wd pull (AdaLN-Zero gates). See
        # experiments.md → E015 for the per-tensor analysis motivating this split.
        use_param_groups = bool(opt_cfg.get("param_groups", False))
        if use_param_groups and weight_decay > 0:
            decay_params, no_decay_params = [], []
            seen = set()
            for name, p in self.named_parameters():
                if not p.requires_grad or id(p) in seen:
                    continue
                seen.add(id(p))
                if (
                    p.ndim < 2                      # biases + LayerNorm γ/β
                    or "to_adaln_zero_gamma" in name
                    or "embed" in name.lower()
                ):
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)
            n_decay = sum(p.numel() for p in decay_params)
            n_no_decay = sum(p.numel() for p in no_decay_params)
            logger.info(
                "configure_optimizers: param_groups split — "
                "decay group: %d tensors / %.2fM params at wd=%.4f; "
                "no-decay group: %d tensors / %.2fM params at wd=0.",
                len(decay_params), n_decay / 1e6, weight_decay,
                len(no_decay_params), n_no_decay / 1e6,
            )
            optimizer = torch.optim.AdamW(
                [
                    {"params": decay_params, "weight_decay": weight_decay},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                lr=opt_cfg.lr,
            )
        else:
            optimizer = torch.optim.AdamW(
                [p for p in self.parameters() if p.requires_grad],
                lr=opt_cfg.lr,
                weight_decay=weight_decay,
            )
        sched_cfg = opt_cfg.get("scheduler", None)
        if sched_cfg is None or sched_cfg.get("name", None) in (None, "none"):
            return optimizer
        name = sched_cfg.name
        if name != "cosine_with_warmup":
            raise ValueError(f"Unknown opt.scheduler.name: {name}")
        warmup = int(sched_cfg.get("warmup_steps", 0))
        total = int(sched_cfg.get("total_steps", 30000))
        min_ratio = float(sched_cfg.get("min_lr_ratio", 0.1))
        import math
        def lr_lambda(step):
            if warmup > 0 and step < warmup:
                return float(step) / float(max(1, warmup))
            progress = (step - warmup) / max(1, total - warmup)
            progress = min(max(progress, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cos
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def on_save_checkpoint(self, checkpoint):
        """Adds additional variables to checkpoint."""
        checkpoint["nflops"] = self.nflops
        checkpoint["nsamples_processed"] = self.nsamples_processed

    def on_load_checkpoint(self, checkpoint):
        """Loads additional variables from checkpoint."""
        try:
            self.nflops = checkpoint["nflops"]
            self.nsamples_processed = checkpoint["nsamples_processed"]
        except:
            logger.info("Failed to load nflops and nsamples_processed from checkpoint")
            self.nflops = 0
            self.nsamples_processed = 0

    def on_before_optimizer_step(self, optimizer):
        if self.global_step % 100 != 0:
            return
        grad_sq = torch.tensor(0.0, device=self.device)
        param_sq = torch.tensor(0.0, device=self.device)
        for p in self.parameters():
            if p.grad is not None:
                grad_sq = grad_sq + p.grad.detach().pow(2).sum()
            if p.requires_grad:
                param_sq = param_sq + p.detach().pow(2).sum()
        self.log(
            "train/grad_norm",
            grad_sq.sqrt(),
            on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True,
        )
        self.log(
            "train/param_norm",
            param_sq.sqrt(),
            on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True,
        )
        for i, pg in enumerate(optimizer.param_groups):
            if "lr" in pg:
                self.log(
                    f"train/lr_pg{i}", float(pg["lr"]),
                    on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True,
                )

    def call_nn(
        self,
        batch: Dict[str, torch.Tensor],
        n_recycle: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Calls NN with recycling. Should this be here or in the NN? Possibly better here,
        in case we want to recycle using decoder for some approach, etc, and this is akin
        to self conditioning, also here.
        Also, if we want to recycle clean sample predictions... Then we'd need this here,
        as the nn does not know about relations between v, x1, ...
        """
        # First call
        nn_out = self.nn(batch)

        # Recycle n_recycle times detaching gradients and updating input
        # Note that recycling is supported by the codebase, but the models provided 
        # with the La-Proteina paper do not use it, nor were trained with it.
        for _ in range(n_recycle):
            x_1_pred = self.fm.nn_out_to_clean_sample_prediction(
                batch=batch, nn_out=nn_out
            )
            batch[f"x_recycle"] = {dm: x_1_pred[dm].detach() for dm in x_1_pred}
            nn_out = self.nn(batch)

        # Final prediction
        return nn_out

    def predict_for_sampling(
        self,
        batch: Dict,
        mode: Literal["full", "ucond"],
        n_recycle: int,
    ) -> Tuple[Union[Dict[str, torch.Tensor], float, None]]:
        """
        This function predicts clean samples for multiple models:
        x_pred, the 'original' model, if mode == full
        x_pred_ucond, the unconditional model, , if mode == ucond

        TODO: Need to update to include autoguidance again

        These predictions will later be used to sample with guidance and autoguidance.

        Args:
            batch: Dict
            mode: str

        Returns:
            x_pred (tensor) for the requested mode
        """
        if mode == "full":
            nn_out = self.call_nn(batch, n_recycle=n_recycle)
        elif mode == "ucond":
            assert "cath_code" in batch or "x_motif" in batch, "Only support CFG when cath_code or x_motif is provided"
            uncond_batch = batch.copy()
            if "cath_code" in uncond_batch:
                uncond_batch.pop("cath_code")
            if "x_motif" in uncond_batch:
                uncond_batch.pop("x_motif")
            nn_out = self.call_nn(uncond_batch, n_recycle=n_recycle)
        else:
            raise IOError(f"Wrong {mode} passed to `predict_for_sampling`")

        return nn_out

    def training_step(self, batch: Dict, batch_idx: int):

        

        """
        Computes training loss for batch of samples.

        Args:
            batch: Data batch.

        Returns:
            Training loss averaged over batch dimension.
        """

        try:

 

            val_step = batch_idx == -1  # validation step is indicated with batch_idx -1
            log_prefix = "validation_loss" if val_step else "train"

            # Add clean samples for all data modes / spaces we are working on
            batch = self.add_clean_samples(batch)


            if self.use_precomputed_latents and self.autoencoder is not None:
                if "local_latents" in batch["x_1"]:
                    # Pass the latent tensor derived from disk to compare with OTF stats.
                    # self.verify_latent_consistency(batch, batch["x_1"]["local_latents"])
                    pass

            # Corrupt the batch
            batch = self.fm.corrupt_batch(batch)  # adds x_1, t, x_0, x_t, mask
            bs, n = batch["mask"].shape

            # Handle conditioning variables
            batch = self.handle_self_cond(
                batch
            )  # self conditioning, adds ["x_sc"] to batch prob 0.5
            batch = self.handle_folding_n_inverse_folding(
                batch
            )  # folding and inverse folding iterations

            # Number of recycling steps
            n_recycle = self.handle_recycling()

            nn_out = self.call_nn(batch, n_recycle=n_recycle)
            losses = self.fm.compute_loss(
                batch=batch,
                nn_out=nn_out,
            )  # Dict[str, Tensor w.batch shape [*]]

            self.log_losses(bs=bs, losses=losses, log_prefix=log_prefix, batch=batch, val_step=val_step)
            train_loss = sum([torch.mean(losses[k]) for k in losses if "_justlog" not in k])

            # Per-length val loss split — distinguishes genuine overfitting from
            # length-specialisation (model fitting one length regime while getting
            # worse elsewhere). losses[k] is shape [b], batch["mask"].sum(-1) is [b].
            if val_step:
                per_protein_loss = sum(
                    losses[k] for k in losses if "_justlog" not in k
                )  # [b]
                lengths = batch["mask"].sum(dim=-1)  # [b]
                for lo, hi in [(50, 175), (175, 300), (300, 425), (425, 513)]:
                    sel = (lengths >= lo) & (lengths < hi)
                    if sel.any():
                        self.log(
                            f"validation_loss_by_len/len_{lo:03d}_{hi:03d}",
                            per_protein_loss[sel].mean(),
                            on_step=False, on_epoch=True, prog_bar=False,
                            logger=True, batch_size=int(sel.sum()),
                            sync_dist=True, add_dataloader_idx=False,
                        )

                # Per-noise-level split. t=1 clean, t=0 pure noise. Bin counts
                # are imbalanced under mix_unif_beta(1.9, 1.0, 0.02) — biased
                # toward t→1; do not population-weight across bins.
                t_per_protein = batch["t"]["bb_ca"]  # [b]
                for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
                    sel = (t_per_protein >= lo) & (t_per_protein < hi)
                    if sel.any():
                        self.log(
                            f"validation_loss_by_t/t_{int(lo*100):03d}_{int(hi*100):03d}",
                            per_protein_loss[sel].mean(),
                            on_step=False, on_epoch=True, prog_bar=False,
                            logger=True, batch_size=int(sel.sum()),
                            sync_dist=True, add_dataloader_idx=False,
                        )

            self.log(
                f"{log_prefix}/loss",
                train_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )



            if not val_step:  # Don't log these for val step
                self.log_train_loss_n_prog_bar(bs, train_loss)
                self.update_n_log_flops(bs, n)
                self.update_n_log_nsamples_processed(bs)
                self.log_nparams()

            return train_loss
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"OOM in batch {batch_idx}. "
                    "Reduce batch_size or accumulate_grad_batches, or reduce K (n_seq/spatial/random neighbors)."
                ) from e
            else:
                raise e

    def add_clean_samples(self, batch: Dict) -> Dict:
        """
        Adds clean sample for all data modes / spaces we are working on. For instance, if we have two
        data modes, bb_ca and local_latents, it adds the clean data to the batch
        x_1 = {
            "bb_ca": Corresponding tensor with clean bb_ca coordinates, shape [b, n, 3]
            "local_latents": Corresponding tensor with clean local_latents, shape [b, n, d]
        }

        Args:
            batch: Batch to add clean samples to.

        Returns:
            Batch with clean sample added.
        """
        batch["x_1"] = {
            dm: self._get_clean_sample(batch, dm)
            for dm in self.cfg_exp.product_flowmatcher
        }
        return batch

    def _get_clean_sample(self, batch: Dict, dm: str) -> torch.Tensor:
        """
        Gets clean sample for a given data mode.

        Args:
            batch: Batch to get clean sample from.
            dm: Data mode to get clean sample for.

        Returns:
            Clean sample for the given data mode.
        """
        if dm == "bb_ca":
            # Check if the coordinates are already stripped to CA only [B, N, 3]
            if batch["coords_nm"].ndim == 3:
                return batch["coords_nm"]
            # Otherwise, extract the CA atom (index 1) from the 37-atom representation [B, N, 37, 3]
            else:
                return batch["coords_nm"][:, :, 1, :]
                
        elif dm == "local_latents":
            if self.use_precomputed_latents:
                assert "mean" in batch and "log_scale" in batch, \
                "Precomputed latents enabled but 'mean' or 'log_scale' not found in batch."
                # Reparameterization trick using precomputed statistics
                mean = batch["mean"]
                log_scale = batch["log_scale"]
                std = torch.exp(log_scale)
                z_latent = mean + std * torch.randn_like(std)
                return z_latent
            else:
                encoded_batch = self.autoencoder.encode(batch)
                return encoded_batch["z_latent"]
        else:
            raise ValueError(f"Loading clean samples from data mode {dm} not supported.")
        
    def handle_self_cond(self, batch: Dict) -> Dict:
        n_recycle = self.cfg_exp.training.get(
            "n_recycle", 0
        )
        if random.random() > 0.5 and self.cfg_exp.training.self_cond:
            with torch.no_grad():
                nn_out = self.call_nn(batch, n_recycle=n_recycle)
                x_1_pred = self.fm.nn_out_to_clean_sample_prediction(
                    batch=batch, nn_out=nn_out
                )
            batch["x_sc"] = {k: x_1_pred[k] for k in x_1_pred}

        return batch

    def handle_recycling(self):
        n_recycle = self.cfg_exp.training.get("n_recycle", 0)
        if n_recycle == 0:
            return 0
        return random.randint(0, n_recycle)  # 0 and n_recycle included

    def handle_folding_n_inverse_folding(self, batch: Dict) -> Dict:
        """
        With 15% probability either a folding or inverse folding iteration.
        If one such iteration (ie 15% of the times), with 50% probability set
        set folding_mode to true, otherwise set inverse_folding_mode to true.

        For inverse folding, we just provide CA.

        Applies to the whole batch.

        With 85% probability sets both to false.

        Adds entries 'folding_mode' and 'inverse_folding_ca_mode' to batch, with
        values being boolean variables (True or False).
        """
        batch["use_ca_coors_nm_feature"] = False
        batch["use_residue_type_feature"] = False
        prob = self.cfg_exp.training.get("p_folding_n_inv_folding_iters", 0.0)
        r1 = random.random()  # float
        if r1 < prob:  # with p=prob
            r2 = random.random()
            if r2 < 0.5:  # with p=0.5
                batch["use_ca_coors_nm_feature"] = True
            else:
                batch["use_residue_type_feature"] = True
        return batch

    def log_losses(
        self,
        bs: int,
        losses: Dict[str, Float[torch.Tensor, "b"]],
        log_prefix: str,
        batch: Dict,
        val_step: bool = False
    ):
        for k in losses:
            log_name = k[: -len("_justlog")] if k.endswith("_justlog") else k

            self.log(
                f"{log_prefix}/loss_{log_name}",
                torch.mean(losses[k]),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )

            if self.cfg_exp.training.get("p_folding_n_inv_folding_iters", 0.0) > 0.0:
                # Log also for folding and inverse folding iters
                # divides by p_aux to account for the fact that for most steps loss will be just zero
                p_aux = self.cfg_exp.training["p_folding_n_inv_folding_iters"] / 2
                loss = torch.mean(losses[k])  # [b]

                f_inv_fold = batch["use_ca_coors_nm_feature"] * 1.0 / p_aux
                self.log(
                    f"{log_prefix}_invfold_ca_iter/loss_{log_name}",
                    loss * f_inv_fold,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

                f_fold = batch["use_residue_type_feature"] * 1.0 / p_aux
                self.log(
                    f"{log_prefix}_fold_iter/loss_{log_name}",
                    loss * f_fold,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

    def log_train_loss_n_prog_bar(self, b: int, train_loss: torch.Tensor):
        self.log(
            f"train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=b,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def log_nparams(self):
        self.log(
            "scaling/nparams",
            self.nparams * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )  # constant line

    def update_n_log_nsamples_processed(self, b: int):
        self.nsamples_processed = self.nsamples_processed + b * self.trainer.world_size
        self.log(
            "scaling/nsamples_processed",
            self.nsamples_processed * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

    def update_n_log_flops(self, b: int, n: int):
        """
        Updates and logs flops, if available
        """
        try:
            nflops_step = self.nn.nflops_computer(
                b, n
            )  # nn should implement this function if we want to see nflops
        except:
            nflops_step = None

        if nflops_step is not None:
            self.nflops = (
                self.nflops + nflops_step * self.trainer.world_size
            )  # Times number of processes so it logs sum across devices
            self.log(
                "scaling/nflops",
                self.nflops * 1.0,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=1,
                sync_dist=True,
            )

    def validation_step(self, batch: Dict, batch_idx: int):
        """
        Validation step.

        Args:
            batch: batch from dataset (see last argument)
            batch_idx: batch index (unused)
        """
        self.validation_step_data(batch, batch_idx)

    def validation_step_data(self, batch: Dict, batch_idx: int):
        """Evaluates the training loss on validation data."""
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx=-1)
            self.validation_output_data.append(loss.item())

    def on_validation_epoch_end(self):
        """
        Takes the samples produced in the validation step, stores them as pdb files, and computes validation metrics.
        It also cleans results.
        """
        self.on_validation_epoch_end_data()

    def on_validation_epoch_end_data(self):
        self.validation_output_data = []

    def configure_inference(self, inf_cfg, nn_ag):
        """Sets inference config with all sampling parameters required by the method (dt, etc)
        and autoguidance network (or None if not provided)."""
        self.inf_cfg = inf_cfg
        self.nn_ag = nn_ag

    def predict_step(self, batch: Dict, batch_idx: int) -> List[Tuple[torch.tensor]]:
        """
        Makes predictions. Should call set_inf_cfg before calling this.

        Args:
            batch: data batch, contains all info for the samples to generate (nsamples, nres, dt, etc.)
                The full batch is passed through to the prediction functions.

        Returns:
            List of tuples. Each tuple represents one of the generated samples, has two elements:
                coordinates tensor of shape [n, 37, 3], and aatype tensor of shape [n].
        """
        # Re-seed per batch so that each protein gets the same initial noise
        # regardless of schedule (different schedules consume different amounts
        # of random state in SDE steps, which otherwise desynchronizes the RNG).
        if hasattr(self, "_generation_base_seed"):
            batch_seed = self._generation_base_seed + batch_idx
            L.seed_everything(batch_seed)

        self_cond = self.inf_cfg.args.self_cond

        nsteps = self.inf_cfg.args.nsteps
        guidance_w = self.inf_cfg.args.get("guidance_w", 1.0)
        ag_ratio = self.inf_cfg.args.get("ag_ratio", 0.0)
        save_trajectory_every = 0

        fn_predict_for_sampling = partial(
            self.predict_for_sampling, n_recycle=self.inf_cfg.get("n_recycle", 0)
        )
        # Fix C2: pull sc_neighbors flags through to the integrator. The bootstrap
        # forward at step 0 only fires when both training-side sc_neighbors=True
        # and inference-side sc_neighbors_bootstrap=True.
        sc_neighbors_active = self.cfg_exp.training.get("sc_neighbors", False)
        sc_neighbors_bootstrap = self.inf_cfg.args.get("sc_neighbors_bootstrap", True)
        gen_samples, extra_info = self.fm.full_simulation(
            batch=batch,
            predict_for_sampling=fn_predict_for_sampling,
            nsteps=nsteps,
            nsamples=batch["nsamples"],
            n=batch["nres"],
            self_cond=self_cond,
            sampling_model_args=self.inf_cfg.model,
            device=self.device,
            save_trajectory_every=save_trajectory_every,
            guidance_w=guidance_w,
            ag_ratio=ag_ratio,
            steering_guide=getattr(self, "steering_guide", None),
            sc_neighbors_active=sc_neighbors_active,
            sc_neighbors_bootstrap=sc_neighbors_bootstrap,
        )
        # Dict with the data_modes as keys, and values with batch shape b
        # extra_info is a dict with additional things, including
        # "mask", whose value is boolean of shape [nsamples, n]

        # Format the generated samples back to proteins
        sample_prots = self.sample_formatting(
            x=gen_samples,
            extra_info=extra_info,
            ret_mode="coors37_n_aatype",
        )
        # Dict with keys `coors` (a37), `residue_type`, and `mask`,
        # shapes [b, n, 37, 3], [b, n], [b, n]

        generation_list = []
        for i in range(sample_prots["coors"].shape[0]):
            generation_list.append(
                (sample_prots["coors"][i], sample_prots["residue_type"][i])
            )  # Tuple (coors [n, 37, 3], aatype [n])
        return generation_list  # List of tupes (coors [n, 37, 3], aatype [n])

    def sample_formatting(
        self,
        x: Dict[str, Tensor],
        extra_info: Dict[str, Tensor],
        ret_mode: str,
    ):
        """
        Given a batch of b samples x produced by the flow matcher, it returns the samples in the requested format (ret_mode).

        Supports `ret_modes` for:
            - `samples` returns the original sample from the flow matcher, a dictionary[str, Tensor].
            for the data modalities, each with batch shape b.
            - `atom37` returns an Tensor of shape [b, n, 37, 3] just for coordinates.
            - `pdb_string` returns a list of dictionaries {"pdb_string": str, "nres": int}, with one dictionary per sample.
            - `coors37_n_aatype` returns a dictionary with keys `coors` (atom37), `residue_type`, and `mask`, and
            values with shapes [b, n, 37, 3] float, [b, n] int, [b, n] boolean, respectively.

        Args:
            x: sample.
            extra_info: a dict with additional things, including:
                - "mask", whose value is boolean of shape [nsamples, n]
                - ...
            ret_mode: target format, for now only supports atom37.

        Returns:
            Sample x in the requested format.
        """
        data_modes = sorted([dm for dm in self.cfg_exp.product_flowmatcher])
        if data_modes == ["bb_ca"]:
            return self._format_sample_bb_ca(
                x=x, ret_mode=ret_mode, mask=extra_info["mask"]
            )
        elif data_modes == ["bb_ca", "local_latents"]:
            return self._format_sample_local_latents(
                x=x, ret_mode=ret_mode, mask=extra_info["mask"]
            )
        else:
            raise NotImplementedError(f"Format {ret_mode} not implemented")

    def _format_sample_bb_ca(
        self,
        x: Dict[str, torch.Tensor],
        ret_mode: str,
        mask: Bool[torch.Tensor, "b n"],
    ):
        if ret_mode == "samples":
            return x

        if ret_mode == "atom37":
            return trans_nm_to_atom37(x["bb_ca"].float())

        elif ret_mode == "coors37_n_aatype":
            coors = (
                ca_nm_to_backbone_atom37(x["bb_ca"].float()) * mask[..., None, None]
            )  # [b, n, 37, 3] — N, CA, C, O reconstructed from Cα trace
            residue_type = torch.zeros_like(coors)[..., 0, 0] * mask  # [b, n]
            return {
                "coors": coors,  # [b, n, 37, 3]
                "residue_type": residue_type.long(),  # [b, n]
                "mask": mask,  # [b, n]
            }

        elif ret_mode == "pdb_string":
            pdb_strings = []

            coors = (
                ca_nm_to_backbone_atom37(x["bb_ca"].float()).detach().cpu().numpy()
            )  # [b, n, 37, 3] — N, CA, C, O reconstructed from Cα trace
            residue_type = np.zeros_like(coors[:, :, 0, 0])  # [b, n]
            atom37_mask = np.zeros_like(coors[:, :, :, 0])  # [b, n, 37]
            atom37_mask[:, :, 0] = 1.0  # N
            atom37_mask[:, :, 1] = 1.0  # CA
            atom37_mask[:, :, 2] = 1.0  # C
            atom37_mask[:, :, 4] = 1.0  # O
            atom37_mask = atom37_mask * mask[..., None]  # [b, n, 37]
            n = coors.shape[-3]

            for i in range(coors.shape[0]):
                prot = create_full_prot(
                    atom37=coors[i, ...],
                    atom37_mask=atom37_mask[i, ...],
                    aatype=residue_type[i, ...],
                )
                pdb_string = to_pdb(prot=prot)
                pdb_strings.append(
                    {
                        "pdb_string": pdb_string,
                        "nres": n,
                    }
                )
            return pdb_strings

        else:
            raise NotImplementedError(
                f"{ret_mode} format for data modes `[bb_ca]` not implemented"
            )

    def _format_sample_local_latents(
        self,
        x: Dict[str, torch.Tensor],
        ret_mode: str,
        mask: Bool[torch.Tensor, "b n"],
    ):
        """
        Given a batch of b samples consisting on `bb_ca` and `local_latents` this
        returns formatted samples.

        Note: This calls the decoder from the autoencoder, since it needs to go from
        local latent variables to the actual coordinates and sequence.

        Note: The self.autoencoder.decode function (used here) returns a dictoinary like
        {
            "coors_nm": [b, n, 37, 3], already masked
            "residue_type": [b, n], already masked, careful with 0s
            "residue_mask": [b, n]
            "atom_mask": [b, n, 37]
        }

        Args:
            x: sample.
            extra_info: a dict with additional things, including:
                - "mask", whose value is boolean of shape [nsamples, n]
                - ...
            ret_mode: target format, for now only supports atom37.

        Returns:
            Sample x in the requested format.
        """
        output_decoder = self.autoencoder.decode(
            z_latent=x["local_latents"], ca_coors_nm=x["bb_ca"], mask=mask
        )

        if ret_mode == "samples":
            return x

        elif ret_mode == "coors37_n_aatype":
            return {
                "coors": nm_to_ang(output_decoder["coors_nm"]),  # [b, n, 37, 3]
                "residue_type": output_decoder["residue_type"],  # [b, n]
                "mask": output_decoder["residue_mask"],  # [b, n]
            }

        elif ret_mode == "pdb_string":
            pdb_strings = []

            coors_atom_37 = (
                nm_to_ang(output_decoder["coors_nm"]).float().detach().cpu().numpy(),
            )  # [b, n, 37, 3]
            residue_type = output_decoder["residue_type"]  # [b, n]
            atom_mask = output_decoder["atom_mask"]  # [b, n, 37]
            n = coors_atom_37.shape[-3]

            for i in range(atom_mask.shape[0]):
                prot = create_full_prot(
                    atom37=coors_atom_37[i, ...],
                    atom37_mask=atom_mask[i, ...],
                    aatype=residue_type[i, ...],
                )
                pdb_string = to_pdb(prot=prot)
                pdb_strings.append(
                    {
                        "pdb_string": pdb_string,
                        "nres": n,
                    }
                )
            return pdb_strings

        else:
            raise NotImplementedError(
                f"{ret_mode} format for data modes `[bb_ca, latent_locals]` not implemented"
            )

