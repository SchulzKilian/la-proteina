"""Training loop for the concept-bottleneck predictor.

Same regime as ``train.py`` (AdamW, warmup+cosine, AMP bf16, length-bucket sampler, z-score
targets, per-property NaN-masked MSE, R² early stopping) plus two bottleneck losses:
  - AA cross-entropy (latent -> amino-acid identity)
  - torsion (sin,cos) MSE, masked by torsion availability
g2 reads g1's *predicted* soft bottleneck (train == inference path); teacher forcing optional.
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import PROPERTY_NAMES, ZScoreStats, LengthBucketBatchSampler
from .cbm_dataset import CBMPropertyDataset, cbm_collate_fn, AA_PAD_IDX
from .cbm_model import CBMPropertyTransformer
from .train import evaluate  # reused unchanged: calls model(latents, mask, t) -> props

logger = logging.getLogger(__name__)


def _bottleneck_losses(aa_logits, tors, batch, device):
    """AA cross-entropy + torsion sin/cos MSE (masked), plus AA accuracy for logging."""
    residue_type = batch["residue_type"].to(device)        # [B,L] (pad = AA_PAD_IDX)
    tors_tgt = batch["torsion_sin_cos"].to(device)         # [B,L,7,2]
    tmask = batch["torsion_mask"].to(device)               # [B,L,7]
    seq_mask = batch["mask"].to(device)                    # [B,L]

    aa_loss = F.cross_entropy(
        aa_logits.reshape(-1, aa_logits.shape[-1]),
        residue_type.reshape(-1),
        ignore_index=AA_PAD_IDX,
    )

    w = (tmask * seq_mask.unsqueeze(-1)).unsqueeze(-1)     # [B,L,7,1]
    tors_loss = (((tors - tors_tgt) ** 2) * w).sum() / w.sum().clamp(min=1.0)

    with torch.no_grad():
        valid = residue_type != AA_PAD_IDX
        pred = aa_logits.argmax(dim=-1)
        aa_acc = ((pred == residue_type) & valid).float().sum() / valid.float().sum().clamp(min=1.0)

    return aa_loss, tors_loss, aa_acc.item()


def train_one_epoch_cbm(model, loader, optimizer, scheduler, scaler, device,
                        lambda_aa, lambda_tors, grad_clip, step_counter, csv_writer=None):
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        latents = batch["latents"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["targets"].to(device)
        t = batch["t"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            props, aa_logits, tors = model(latents, mask, t, return_aux=True)
            valid = ~torch.isnan(targets)
            diff = ((props - targets) ** 2) * valid.float()
            per_prop_mse = diff.sum(dim=0) / valid.float().sum(dim=0).clamp(min=1.0)
            prop_loss = per_prop_mse.mean()
            aa_loss, tors_loss, aa_acc = _bottleneck_losses(aa_logits, tors, batch, device)
            loss = prop_loss + lambda_aa * aa_loss + lambda_tors * tors_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        step_counter += 1
        total_loss += loss.item()
        n_batches += 1
        if csv_writer is not None:
            row = {
                "step": step_counter, "total_loss": loss.item(),
                "prop_loss": prop_loss.item(), "aa_loss": aa_loss.item(),
                "tors_loss": tors_loss.item(), "aa_acc": aa_acc,
                "lr": scheduler.get_last_lr()[0],
            }
            for i, p in enumerate(PROPERTY_NAMES):
                row[f"mse_{p}"] = per_prop_mse[i].item()
            csv_writer.writerow(row)

    return total_loss / max(n_batches, 1), step_counter


def train_fold_cbm(fold_idx, train_records, val_records, prop_df, output_dir, config, device):
    logger.info("=== CBM Fold %d ===", fold_idx)

    # z-score fit on training targets (identical recipe to train.py)
    prop_indexed = prop_df.set_index("protein_id")
    train_pids = [r.protein_id for r in train_records]
    available = [p for p in PROPERTY_NAMES if p in prop_indexed.columns]
    tt = prop_indexed.loc[prop_indexed.index.isin(train_pids), available].reindex(train_pids).values.astype(np.float32)
    if len(available) < len(PROPERTY_NAMES):
        full = np.full((len(train_pids), len(PROPERTY_NAMES)), np.nan, dtype=np.float32)
        for i, p in enumerate(PROPERTY_NAMES):
            if p in available:
                full[:, i] = tt[:, available.index(p)]
        tt = full
    stats = ZScoreStats.fit(tt)

    train_ds = CBMPropertyDataset(train_records, prop_df, stats=stats)
    val_ds = CBMPropertyDataset(val_records, prop_df, stats=stats)

    bs = config.get("batch_size", 16)
    train_sampler = LengthBucketBatchSampler([r.length for r in train_records], bs, shuffle=True, seed=42 + fold_idx)
    val_sampler = LengthBucketBatchSampler([r.length for r in val_records], bs, shuffle=False)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=cbm_collate_fn,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=cbm_collate_fn,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    model = CBMPropertyTransformer(
        latent_dim=config.get("latent_dim", 8), d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4), n_layers=config.get("n_layers", 3),
        ffn_expansion=config.get("ffn_expansion", 4), dropout=config.get("dropout", 0.1),
        n_properties=len(PROPERTY_NAMES), max_len=config.get("max_len", 1024),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 3e-4),
                                  weight_decay=config.get("weight_decay", 0.01))
    max_epochs = config.get("max_epochs", 30)
    warmup = config.get("warmup_steps", 500)
    total_steps = max_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    curves_path = output_dir / "training_curves.csv"
    curves_file = open(curves_path, "a", newline="")
    step_fields = ["fold", "step", "total_loss", "prop_loss", "aa_loss", "tors_loss", "aa_acc", "lr"] + [f"mse_{p}" for p in PROPERTY_NAMES]
    raw_writer = csv.DictWriter(curves_file, fieldnames=step_fields)
    if fold_idx == 0:
        raw_writer.writeheader()

    class FoldWriter:
        def __init__(self, w, f): self.w, self.f = w, f
        def writerow(self, row): row["fold"] = self.f; self.w.writerow(row)
    step_writer = FoldWriter(raw_writer, fold_idx)

    epoch_path = output_dir / "epoch_metrics.csv"
    epoch_file = open(epoch_path, "a", newline="")
    epoch_fields = ["fold", "epoch", "train_loss", "r2_mean"] + [f"r2_{p}" for p in PROPERTY_NAMES]
    epoch_writer = csv.DictWriter(epoch_file, fieldnames=epoch_fields)
    if fold_idx == 0:
        epoch_writer.writeheader()

    best_r2, patience_ctr, step_counter = -float("inf"), 0, 0
    patience = config.get("patience", 5)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"fold_{fold_idx}_best.pt"

    for epoch in range(max_epochs):
        t0 = time.time()
        train_loss, step_counter = train_one_epoch_cbm(
            model, train_loader, optimizer, scheduler, scaler, device,
            lambda_aa=config.get("lambda_aa", 1.0), lambda_tors=config.get("lambda_tors", 1.0),
            grad_clip=config.get("grad_clip", 1.0), step_counter=step_counter, csv_writer=step_writer,
        )
        curves_file.flush()
        val_results = evaluate(model, val_loader, device, stats)
        val_r2 = val_results["r2_mean"]
        logger.info("CBM Fold %d Epoch %d: train_loss=%.4f val_r2_mean=%.4f (%.1fs)",
                    fold_idx, epoch, train_loss, val_r2, time.time() - t0)

        row = {"fold": fold_idx, "epoch": epoch, "train_loss": train_loss, "r2_mean": val_r2}
        for p in PROPERTY_NAMES:
            row[f"r2_{p}"] = val_results.get(f"r2_{p}", float("nan"))
        epoch_writer.writerow(row)
        epoch_file.flush()

        if val_r2 > best_r2:
            best_r2, patience_ctr = val_r2, 0
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "val_r2_mean": val_r2, "val_results": val_results,
                "stats_mean": stats.mean, "stats_std": stats.std,
                "cbm": True,
            }, ckpt_path)
            logger.info("  New best R²=%.4f, saved", best_r2)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info("  Early stopping at epoch %d", epoch)
                break

    curves_file.close()
    epoch_file.close()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    best_val = evaluate(model, val_loader, device, stats)
    best_val["fold"] = fold_idx
    best_val["best_epoch"] = ckpt["epoch"]
    logger.info("CBM Fold %d best: epoch=%d r2_mean=%.4f", fold_idx, ckpt["epoch"], best_val["r2_mean"])
    return best_val
