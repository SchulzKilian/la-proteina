"""Training loop and evaluation for multi-task property predictor."""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

from .dataset import (
    PROPERTY_NAMES,
    PropertyDataset,
    ZScoreStats,
    LengthBucketBatchSampler,
    collate_fn,
)
from .model import PropertyTransformer

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: PropertyTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_clip: float = 1.0,
    step_counter: int = 0,
    csv_writer=None,
) -> tuple[float, int]:
    """Train for one epoch. Returns (mean_loss, updated_step_counter)."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        latents = batch["latents"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["targets"].to(device)
        t = batch["t"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            preds = model(latents, mask, t)  # [B, 13]
            # Per-property MSE, handling NaN targets
            valid = ~torch.isnan(targets)
            diff = (preds - targets) ** 2
            diff = diff * valid.float()
            # Mean over properties, then over batch
            per_prop_mse = diff.sum(dim=0) / valid.float().sum(dim=0).clamp(min=1.0)
            loss = per_prop_mse.mean()

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
                "step": step_counter,
                "total_loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
            }
            for i, pname in enumerate(PROPERTY_NAMES):
                row[f"mse_{pname}"] = per_prop_mse[i].item()
            csv_writer.writerow(row)

    return total_loss / max(n_batches, 1), step_counter


@torch.no_grad()
def evaluate(
    model: PropertyTransformer,
    loader: DataLoader,
    device: torch.device,
    stats: ZScoreStats,
) -> dict[str, float]:
    """Evaluate on a dataset. Returns per-property R² (de-normalised) and mean R²."""
    model.eval()

    all_preds = []
    all_targets = []

    for batch in loader:
        latents = batch["latents"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["targets"]  # [B, 13] z-scored
        t = batch["t"].to(device)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            preds = model(latents, mask, t)  # [B, 13]

        all_preds.append(preds.float().cpu().numpy())
        all_targets.append(targets.numpy())

    preds_arr = np.concatenate(all_preds, axis=0)   # [N, 13]
    targets_arr = np.concatenate(all_targets, axis=0)  # [N, 13]

    # De-normalise
    preds_denorm = stats.inverse_transform(preds_arr)
    targets_denorm = stats.inverse_transform(targets_arr)

    results = {}
    r2_values = []
    for i, pname in enumerate(PROPERTY_NAMES):
        valid = ~np.isnan(targets_denorm[:, i])
        if valid.sum() < 2:
            results[f"r2_{pname}"] = float("nan")
            continue
        r2 = r2_score(targets_denorm[valid, i], preds_denorm[valid, i])
        results[f"r2_{pname}"] = r2
        r2_values.append(r2)

    results["r2_mean"] = float(np.mean(r2_values)) if r2_values else float("nan")
    return results


def train_fold(
    fold_idx: int,
    train_records: list,
    val_records: list,
    prop_df: pd.DataFrame,
    output_dir: Path,
    config: dict,
    device: torch.device,
) -> dict:
    """Train one fold to convergence with early stopping.

    Returns dict with per-property R² on validation set.
    """
    logger.info("=== Fold %d ===", fold_idx)

    # Fit z-score on training data — use indexed lookup for speed
    prop_indexed = prop_df.set_index("protein_id")
    train_pids = [r.protein_id for r in train_records]
    available_props = [p for p in PROPERTY_NAMES if p in prop_indexed.columns]
    train_targets = prop_indexed.loc[
        prop_indexed.index.isin(train_pids), available_props
    ].reindex(train_pids).values.astype(np.float32)
    # Add NaN columns for missing properties
    if len(available_props) < len(PROPERTY_NAMES):
        full_targets = np.full((len(train_pids), len(PROPERTY_NAMES)), np.nan, dtype=np.float32)
        for i, p in enumerate(PROPERTY_NAMES):
            if p in available_props:
                full_targets[:, i] = train_targets[:, available_props.index(p)]
        train_targets = full_targets
    stats = ZScoreStats.fit(train_targets)

    train_ds = PropertyDataset(train_records, prop_df, stats=stats)
    val_ds = PropertyDataset(val_records, prop_df, stats=stats)

    batch_size = config.get("batch_size", 16)
    train_sampler = LengthBucketBatchSampler(
        [r.length for r in train_records],
        batch_size=batch_size,
        shuffle=True,
        seed=42 + fold_idx,
    )
    val_sampler = LengthBucketBatchSampler(
        [r.length for r in val_records],
        batch_size=batch_size,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=collate_fn,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler, collate_fn=collate_fn,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    model = PropertyTransformer(
        latent_dim=config.get("latent_dim", 8),
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 3),
        ffn_expansion=config.get("ffn_expansion", 4),
        dropout=config.get("dropout", 0.1),
        n_properties=len(PROPERTY_NAMES),
        max_len=config.get("max_len", 1024),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 3e-4),
        weight_decay=config.get("weight_decay", 0.01),
    )

    max_epochs = config.get("max_epochs", 30)
    warmup_steps = config.get("warmup_steps", 500)
    total_steps = max_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training curves CSV
    curves_path = output_dir / "training_curves.csv"
    curves_file = open(curves_path, "a", newline="")
    step_fields = ["step", "total_loss", "lr"] + [f"mse_{p}" for p in PROPERTY_NAMES]
    step_writer = csv.DictWriter(curves_file, fieldnames=["fold"] + step_fields)
    if fold_idx == 0:
        step_writer.writeheader()

    class FoldStepWriter:
        """Wraps csv writer to inject fold column."""
        def __init__(self, writer, fold):
            self.writer = writer
            self.fold = fold
        def writerow(self, row):
            row["fold"] = self.fold
            self.writer.writerow(row)

    fold_step_writer = FoldStepWriter(step_writer, fold_idx)

    # Epoch-level log
    epoch_log_path = output_dir / "epoch_metrics.csv"
    epoch_fields = ["fold", "epoch", "train_loss", "r2_mean"] + [f"r2_{p}" for p in PROPERTY_NAMES]
    epoch_file = open(epoch_log_path, "a", newline="")
    epoch_writer = csv.DictWriter(epoch_file, fieldnames=epoch_fields)
    if fold_idx == 0:
        epoch_writer.writeheader()

    best_r2 = -float("inf")
    patience_counter = 0
    patience = config.get("patience", 5)
    step_counter = 0
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"fold_{fold_idx}_best.pt"

    for epoch in range(max_epochs):
        t0 = time.time()
        train_loss, step_counter = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            grad_clip=config.get("grad_clip", 1.0),
            step_counter=step_counter,
            csv_writer=fold_step_writer,
        )
        curves_file.flush()

        val_results = evaluate(model, val_loader, device, stats)
        val_r2_mean = val_results["r2_mean"]
        elapsed = time.time() - t0

        logger.info(
            "Fold %d Epoch %d: train_loss=%.4f val_r2_mean=%.4f (%.1fs)",
            fold_idx, epoch, train_loss, val_r2_mean, elapsed,
        )

        # Log epoch
        epoch_row = {"fold": fold_idx, "epoch": epoch, "train_loss": train_loss, "r2_mean": val_r2_mean}
        for p in PROPERTY_NAMES:
            epoch_row[f"r2_{p}"] = val_results.get(f"r2_{p}", float("nan"))
        epoch_writer.writerow(epoch_row)
        epoch_file.flush()

        # Early stopping
        if val_r2_mean > best_r2:
            best_r2 = val_r2_mean
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_r2_mean": val_r2_mean,
                "val_results": val_results,
                "stats_mean": stats.mean,
                "stats_std": stats.std,
            }, ckpt_path)
            logger.info("  New best R²=%.4f, saved checkpoint", best_r2)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("  Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    curves_file.close()
    epoch_file.close()

    # Load best checkpoint and evaluate
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    best_val = evaluate(model, val_loader, device, stats)
    best_val["fold"] = fold_idx
    best_val["best_epoch"] = ckpt["epoch"]

    logger.info("Fold %d best: epoch=%d r2_mean=%.4f", fold_idx, ckpt["epoch"], best_val["r2_mean"])
    return best_val
