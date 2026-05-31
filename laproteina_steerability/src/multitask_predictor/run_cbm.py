"""CLI entry point for the concept-bottleneck property predictor.

Mirrors run.py exactly (same data, splits, hyperparameters, 5-fold ensemble) — the only
differences are the CBM model + AA/torsion bottleneck losses. Trains on the FULL
processed_latents_300_800 set (subsample forced off) to match the no-bottleneck NA-v1 predictor.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.part2_property_probes.properties import load_properties, align_properties_to_latents
from src.multitask_predictor.dataset import PROPERTY_NAMES, create_held_out_split, create_fold_assignments
from src.multitask_predictor.cbm_dataset import load_cbm_dataset
from src.multitask_predictor.cbm_train import train_fold_cbm

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="CBM multi-task property predictor (t=1)")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--folds", type=str, default=None, help="e.g. '0' or '0,1,2,3,4'")
    parser.add_argument("--lambda-aa", type=float, default=1.0)
    parser.add_argument("--lambda-tors", type=float, default=1.0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("logs/multitask_cbm") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = {
        "latent_dim": 8, "d_model": 128, "n_heads": 4, "n_layers": 3,
        "ffn_expansion": 4, "dropout": 0.1, "lr": 3e-4, "weight_decay": 0.01,
        "grad_clip": 1.0, "warmup_steps": 500,
        "max_epochs": 2 if args.smoke_test else args.max_epochs,
        "patience": 5, "batch_size": args.batch_size, "max_len": 1024, "seed": 42,
        "lambda_aa": args.lambda_aa, "lambda_tors": args.lambda_tors,
        "data": cfg["data"], "property_file": cfg["part2"]["property_file"],
        "property_names": cfg["part2"]["property_names"], "timestamp": timestamp, "cbm": True,
    }
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(train_config, f, default_flow_style=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    latent_dir = cfg["data"]["latent_dir"]
    if not Path(latent_dir).is_dir():
        fallback = Path(__file__).resolve().parents[2] / ".." / "data" / "pdb_train" / "processed_latents_300_800"
        latent_dir = str(fallback.resolve())
    logger.info("Loading latents (+AA/torsion bottleneck targets) from %s ...", latent_dir)

    length_range = tuple(cfg["data"]["length_range"]) if cfg["data"].get("length_range") else None
    records = load_cbm_dataset(latent_dir, length_range=length_range)  # FULL set (no subsample)
    logger.info("Loaded %d CBM records", len(records))

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
    logger.info("After alignment: %d proteins", len(records))

    train_ids, test_ids = create_held_out_split(records, test_fraction=0.1, seed=42)
    with open(output_dir / "heldout_test_ids.txt", "w") as f:
        for pid in test_ids:
            f.write(pid + "\n")
    fold_df = create_fold_assignments(train_ids, n_folds=5)
    fold_df.to_csv(output_dir / "fold_assignments.csv", index=False)
    logger.info("Held-out: %d, Train: %d", len(test_ids), len(train_ids))

    record_by_id = {r.protein_id: r for r in records}

    if args.smoke_test:
        fold_indices = [0]
    elif args.folds is not None:
        fold_indices = [int(x) for x in args.folds.split(",")]
    else:
        fold_indices = list(range(5))

    all_results = []
    for fold_idx in fold_indices:
        val_ids = set(fold_df[fold_df["fold"] == fold_idx]["protein_id"])
        tr_ids = set(fold_df[fold_df["fold"] != fold_idx]["protein_id"])
        tr_recs = [record_by_id[p] for p in sorted(tr_ids) if p in record_by_id]
        val_recs = [record_by_id[p] for p in sorted(val_ids) if p in record_by_id]
        logger.info("Fold %d: %d train, %d val", fold_idx, len(tr_recs), len(val_recs))
        result = train_fold_cbm(fold_idx, tr_recs, val_recs, prop_df, output_dir, train_config, device)
        all_results.append(result)

    pd.DataFrame(all_results).to_csv(output_dir / "results_per_fold.csv", index=False)

    if len(all_results) > 1:
        rows = []
        for p in PROPERTY_NAMES:
            col = f"r2_{p}"
            vals = [r[col] for r in all_results if not np.isnan(r.get(col, np.nan))]
            rows.append({"property": p, "r2_mean": np.mean(vals) if vals else np.nan,
                         "r2_std": np.std(vals) if vals else np.nan, "n_folds": len(vals)})
        mv = [r["r2_mean"] for r in all_results if not np.isnan(r.get("r2_mean", np.nan))]
        rows.append({"property": "MEAN", "r2_mean": np.mean(mv) if mv else np.nan,
                     "r2_std": np.std(mv) if mv else np.nan, "n_folds": len(mv)})
        pd.DataFrame(rows).to_csv(output_dir / "results_summary.csv", index=False)

    logger.info("All outputs in %s", output_dir)
    return output_dir


if __name__ == "__main__":
    main()
