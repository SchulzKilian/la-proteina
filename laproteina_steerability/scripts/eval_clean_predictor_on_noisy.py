"""Apples-to-apples noise-aware vs clean predictor comparison.

Question: how well does the clean predictor extract the deterministic signal
that survives in the noisy latent (z_t at t∈[0.3, 0.8])? Compared to the
noise-aware predictor scored on the same noisy inputs?

The noise-aware predictor's saved `r2_*_noisy` values use the same noise
injection scheme. This script applies the same scheme to the clean predictor
and reports per-property R² on the same val split for each of the 5 folds.

Output: results_clean_on_noisy.csv (per-property R² × 5 folds).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Set up paths
_ROOT = Path('/home/ks2218/la-proteina/laproteina_steerability')
sys.path.insert(0, str(_ROOT))

from src.data.loader import load_dataset
from src.multitask_predictor.dataset import (
    PROPERTY_NAMES,
    PropertyDataset,
    ZScoreStats,
    LengthBucketBatchSampler,
    collate_fn,
)
from src.multitask_predictor.model import PropertyTransformer
from src.multitask_predictor.train import evaluate
from src.part2_property_probes.properties import (
    load_properties,
    align_properties_to_latents,
)
from scripts.add_noisy_latents import NoisyPropertyDataset, _worker_init_fn

CLEAN_DIR = _ROOT / "logs/multitask_t1/20260427_161809"
NA_DIR = _ROOT / "logs/multitask_t1_noise_aware/20260505_110348"
OUT_CSV = Path('/home/ks2218/la-proteina/results/clean_predictor_on_noisy_r2.csv')


def main():
    # Match the noise-aware training config exactly: same t_min/t_max, sigma_langevin, batch size.
    na_cfg = yaml.safe_load((NA_DIR / "config.yaml").read_text())
    print(f"noise-aware config: t_min={na_cfg['t_min']}, t_max={na_cfg['t_max']}, sigma_langevin={na_cfg['sigma_langevin']}, batch_size={na_cfg['batch_size']}")
    print()

    # Match the clean training data config too — same source records.
    clean_cfg = yaml.safe_load((CLEAN_DIR / "config.yaml").read_text())

    # Load data the same way the original training did.
    data_cfg = clean_cfg["data"]
    records = load_dataset(
        latent_dir=data_cfg["latent_dir"],
        file_format=data_cfg["file_format"],
        field_names=data_cfg["field_names"],
        length_range=tuple(data_cfg["length_range"]) if data_cfg.get("length_range") else None,
        subsample=data_cfg.get("subsample"),
    )
    prop_df = load_properties(
        property_file=clean_cfg["property_file"],
        property_names=clean_cfg["property_names"],
        property_granularity={},  # all-protein granularity is the default
    )
    records, prop_df = align_properties_to_latents(
        records, prop_df,
        property_names=clean_cfg["property_names"],
        property_granularity={},
    )
    fold_assignments = pd.read_csv(CLEAN_DIR / "fold_assignments.csv")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Total records: {len(records)}")
    print(f"Properties: {len(PROPERTY_NAMES)}")
    print()

    rows = []
    for fold_idx in range(5):
        print(f"=== Fold {fold_idx} ===")
        # The val set for this fold = proteins assigned to this fold.
        val_ids = set(fold_assignments[fold_assignments.fold == fold_idx].protein_id)
        val_records = [r for r in records if r.protein_id in val_ids]
        print(f"  val proteins: {len(val_records)}")

        # Load clean predictor checkpoint
        ckpt_path = CLEAN_DIR / f"checkpoints/fold_{fold_idx}_best.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        stats = ZScoreStats(mean=ckpt["stats_mean"], std=ckpt["stats_std"])

        # Build noisy validation set with SAME seed as noise-aware training used for val.
        # noise-aware used `seed=43 + fold_idx * 1000` for val noise.
        val_ds = NoisyPropertyDataset(
            val_records, prop_df, stats=stats,
            t_min=na_cfg["t_min"], t_max=na_cfg["t_max"],
            sigma_langevin=na_cfg["sigma_langevin"],
            seed=43 + fold_idx * 1000,
        )
        sampler = LengthBucketBatchSampler(
            [r.length for r in val_records], batch_size=na_cfg["batch_size"], shuffle=False,
        )
        loader = DataLoader(
            val_ds, batch_sampler=sampler, collate_fn=collate_fn,
            num_workers=2, pin_memory=True, persistent_workers=True,
            worker_init_fn=_worker_init_fn,
        )

        # Build clean predictor model and load weights
        model = PropertyTransformer(
            latent_dim=clean_cfg["latent_dim"],
            d_model=clean_cfg["d_model"],
            n_heads=clean_cfg["n_heads"],
            n_layers=clean_cfg["n_layers"],
            ffn_expansion=clean_cfg["ffn_expansion"],
            dropout=clean_cfg["dropout"],
            n_properties=len(PROPERTY_NAMES),
            max_len=clean_cfg["max_len"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Evaluate
        metrics = evaluate(model, loader, device, stats)
        # Save per-property R²
        for prop in PROPERTY_NAMES:
            key = f"r2_{prop}"
            r2 = float(metrics.get(key, float("nan")))
            rows.append({"fold": fold_idx, "property": prop, "r2_clean_on_noisy": r2})
        mean_r2 = float(np.mean([metrics[f"r2_{p}"] for p in PROPERTY_NAMES if f"r2_{p}" in metrics]))
        print(f"  r2_mean (clean on noisy) = {mean_r2:.4f}")
        print()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
