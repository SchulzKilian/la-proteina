"""Capacity probing sweep: fit probes of increasing size to La-Proteina latents
and measure V-information (via achieved R²) per property.

Usage:
    python -m src.capacity_probing.run_sweep --config config/default.yaml --folds 0

What it does:
    1. Loads latents + properties *once* (the expensive step, ~45 min on Lustre).
    2. For each probe architecture in PROBE_CONFIGS (7 sizes, no Transformer —
       the 3L-Tx data already exists in multitask_t1 runs), trains to convergence
       on the selected fold(s).
    3. Writes per-probe best R² + per-epoch training curves to a single output CSV.
    4. Writes a summary table: rows = probes (ordered by #params), columns = 13 properties.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

# Path to steerability root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_dataset  # noqa: E402
from src.part2_property_probes.properties import (  # noqa: E402
    load_properties,
    align_properties_to_latents,
)
from src.multitask_predictor.dataset import (  # noqa: E402
    PROPERTY_NAMES,
    PropertyDataset,
    ZScoreStats,
    LengthBucketBatchSampler,
    collate_fn,
    create_held_out_split,
    create_fold_assignments,
)
from src.capacity_probing.probes import PROBE_CONFIGS, build_probe  # noqa: E402

logger = logging.getLogger(__name__)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_probe(
    probe: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    stats: ZScoreStats,
    device: torch.device,
    max_epochs: int = 20,
    patience: int = 4,
    lr: float = 3e-3,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
) -> dict:
    """Train a single probe; return best R² + epoch-by-epoch curves."""
    probe = probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    best_r2_mean = -float("inf")
    best_epoch = -1
    best_per_prop = None
    no_improve = 0
    curves = []

    for epoch in range(max_epochs):
        t0 = time.time()
        probe.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            z = batch["latents"].to(device)
            mask = batch["mask"].to(device)
            tgt = batch["targets"].to(device)
            opt.zero_grad(set_to_none=True)
            preds = probe(z, mask)
            valid = ~torch.isnan(tgt)
            sq = (preds - tgt) ** 2
            sq = sq * valid.float()
            per_prop_mse = sq.sum(dim=0) / valid.float().sum(dim=0).clamp(min=1.0)
            loss = per_prop_mse.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(probe.parameters(), grad_clip)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        mean_train_loss = total_loss / max(n_batches, 1)

        # Eval
        probe.eval()
        Ps, Ts = [], []
        with torch.no_grad():
            for batch in val_loader:
                z = batch["latents"].to(device)
                mask = batch["mask"].to(device)
                tgt = batch["targets"]
                p = probe(z, mask).cpu().numpy()
                Ps.append(p)
                Ts.append(tgt.numpy())
        P = np.concatenate(Ps); T = np.concatenate(Ts)
        P_dn = stats.inverse_transform(P); T_dn = stats.inverse_transform(T)
        per_prop_r2 = {}
        vals = []
        for i, name in enumerate(PROPERTY_NAMES):
            v = ~np.isnan(T_dn[:, i])
            if v.sum() < 2:
                per_prop_r2[name] = float("nan")
                continue
            r2 = r2_score(T_dn[v, i], P_dn[v, i])
            per_prop_r2[name] = r2
            vals.append(r2)
        r2_mean = float(np.mean(vals)) if vals else float("nan")
        elapsed = time.time() - t0
        curves.append({
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "r2_mean": r2_mean,
            "elapsed_s": elapsed,
            **{f"r2_{k}": v for k, v in per_prop_r2.items()},
        })
        logger.info(
            "  epoch %d: train_loss=%.4f r2_mean=%.4f (%.1fs)",
            epoch, mean_train_loss, r2_mean, elapsed,
        )

        if r2_mean > best_r2_mean:
            best_r2_mean = r2_mean
            best_epoch = epoch
            best_per_prop = per_prop_r2
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("  early stop at epoch %d", epoch)
                break

    return {
        "best_r2_mean": best_r2_mean,
        "best_epoch": best_epoch,
        "best_per_prop": best_per_prop,
        "curves": curves,
    }


def main():
    parser = argparse.ArgumentParser(description="Capacity probing sweep")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--folds", type=str, default="0",
                        help="Comma-separated fold indices (default: '0'; pass '0,2' for 2 folds)")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("logs/capacity_probing") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Outputs: %s", out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Load data once ---
    latent_dir = cfg["data"]["latent_dir"]
    if not Path(latent_dir).is_dir():
        fallback = Path(__file__).resolve().parents[2] / ".." / "data" / "pdb_train" / "processed_latents_300_800"
        if fallback.is_dir():
            latent_dir = str(fallback.resolve())
            logger.info("Using fallback latent dir: %s", latent_dir)

    logger.info("Loading latents from %s ...", latent_dir)
    rng = np.random.default_rng(42)
    records = load_dataset(
        latent_dir=latent_dir,
        file_format=cfg["data"]["file_format"],
        field_names=cfg["data"]["field_names"],
        load_coords=False,
        subsample=cfg["data"].get("subsample"),
        rng=rng,
        length_range=tuple(cfg["data"]["length_range"]) if cfg["data"].get("length_range") else None,
    )
    logger.info("Loaded %d records", len(records))

    prop_df = load_properties(
        property_file=cfg["part2"]["property_file"],
        property_names=cfg["part2"]["property_names"],
        property_granularity=cfg["part2"]["property_granularity"],
    )
    records, prop_df = align_properties_to_latents(
        records, prop_df,
        property_names=cfg["part2"]["property_names"],
        property_granularity=cfg["part2"]["property_granularity"],
    )
    logger.info("After alignment: %d records", len(records))

    train_ids, test_ids = create_held_out_split(records, test_fraction=0.1, seed=42)
    fold_df = create_fold_assignments(train_ids, n_folds=5)
    record_by_id = {r.protein_id: r for r in records}

    fold_indices = [int(x) for x in args.folds.split(",")]
    logger.info("Running probe sweep on folds: %s", fold_indices)

    # --- Sweep ---
    all_rows = []
    curves_rows = []
    for fold_idx in fold_indices:
        logger.info("=== Fold %d ===", fold_idx)
        val_ids = set(fold_df[fold_df["fold"] == fold_idx]["protein_id"])
        fold_train_ids = set(fold_df[fold_df["fold"] != fold_idx]["protein_id"])
        fold_train_records = [record_by_id[pid] for pid in sorted(fold_train_ids) if pid in record_by_id]
        fold_val_records = [record_by_id[pid] for pid in sorted(val_ids) if pid in record_by_id]

        # Fit z-score on training targets
        prop_indexed = prop_df.set_index("protein_id")
        available = [p for p in PROPERTY_NAMES if p in prop_indexed.columns]
        tgt_train = prop_indexed.loc[
            prop_indexed.index.isin([r.protein_id for r in fold_train_records]),
            available,
        ].reindex([r.protein_id for r in fold_train_records]).values.astype(np.float32)
        if len(available) < len(PROPERTY_NAMES):
            full = np.full((len(fold_train_records), len(PROPERTY_NAMES)), np.nan, dtype=np.float32)
            for i, p in enumerate(PROPERTY_NAMES):
                if p in available:
                    full[:, i] = tgt_train[:, available.index(p)]
            tgt_train = full
        stats = ZScoreStats.fit(tgt_train)

        train_ds = PropertyDataset(fold_train_records, prop_df, stats=stats)
        val_ds = PropertyDataset(fold_val_records, prop_df, stats=stats)
        train_sampler = LengthBucketBatchSampler(
            [r.length for r in fold_train_records],
            batch_size=args.batch_size,
            shuffle=True,
            seed=42 + fold_idx,
        )
        val_sampler = LengthBucketBatchSampler(
            [r.length for r in fold_val_records],
            batch_size=args.batch_size,
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

        for probe_cfg in PROBE_CONFIGS:
            probe = build_probe(probe_cfg)
            n_params = count_params(probe)
            logger.info("--- %s (%d params), fold %d ---", probe_cfg["name"], n_params, fold_idx)
            result = train_one_probe(
                probe, train_loader, val_loader, stats, device,
                max_epochs=args.max_epochs, patience=args.patience, lr=args.lr,
            )
            row = {
                "probe": probe_cfg["name"],
                "n_params": n_params,
                "fold": fold_idx,
                "best_epoch": result["best_epoch"],
                "best_r2_mean": result["best_r2_mean"],
                **{f"r2_{k}": v for k, v in (result["best_per_prop"] or {}).items()},
            }
            all_rows.append(row)
            for c in result["curves"]:
                c_row = {"probe": probe_cfg["name"], "n_params": n_params, "fold": fold_idx, **c}
                curves_rows.append(c_row)

            # Persist after every probe so nothing is lost on early exit
            pd.DataFrame(all_rows).to_csv(out_dir / "sweep_best.csv", index=False)
            pd.DataFrame(curves_rows).to_csv(out_dir / "sweep_curves.csv", index=False)

    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
