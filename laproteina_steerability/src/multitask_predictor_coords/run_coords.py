"""CLI entry point: 5-fold from-scratch training of the CA-conditioned predictor.

Mirrors `multitask_predictor.run` so the resulting log dir / fold-assignments
layout matches `logs/multitask_t1/<timestamp>/`. Outputs land under
`logs/multitask_t1_coords/<timestamp>/`.
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

from src.data.loader import load_dataset
from src.part2_property_probes.properties import (
    load_properties,
    align_properties_to_latents,
)
from src.multitask_predictor_coords.dataset_coords import (
    PROPERTY_NAMES,
    create_held_out_split,
    create_fold_assignments,
)
from src.multitask_predictor_coords.train_coords import train_fold

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="CA-conditioned multi-task property predictor (t=1 screening)")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--folds", type=str, default=None,
                        help="Comma-separated fold indices to run (e.g. '0' or '0,1,2,3,4')")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick smoke test: 2 epochs, fold 0 only")
    parser.add_argument("--out-root", type=Path,
                        default=Path("logs/multitask_t1_coords"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root if args.out_root.is_absolute() else Path.cwd() / args.out_root
    output_dir = out_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = {
        "latent_dim": 8,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 3,
        "ffn_expansion": 4,
        "dropout": 0.1,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "warmup_steps": 500,
        "max_epochs": 2 if args.smoke_test else args.max_epochs,
        "patience": 5,
        "batch_size": args.batch_size,
        "max_len": 1024,
        # CoordPairBias
        "n_rbf": 32,
        "rbf_max_nm": 8.0,
        "relpos_clamp": 32,
        "relpos_dim": 32,
        "seed": 42,
        "data": cfg["data"],
        "property_file": cfg["part2"]["property_file"],
        "property_names": cfg["part2"]["property_names"],
        "timestamp": timestamp,
    }
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(train_config, f, default_flow_style=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    latent_dir = cfg["data"]["latent_dir"]
    if not Path(latent_dir).is_dir():
        fallback = Path(__file__).resolve().parents[2] / ".." / "data" / "pdb_train" / "processed_latents_300_800"
        if fallback.is_dir():
            latent_dir = str(fallback.resolve())
            logger.info("Primary latent dir not found, using fallback: %s", latent_dir)
        else:
            raise FileNotFoundError(
                f"Latent dir not found at {cfg['data']['latent_dir']} or {fallback}"
            )
    logger.info("Loading latents (with CA coords) from %s ...", latent_dir)
    rng = np.random.default_rng(42)
    records = load_dataset(
        latent_dir=latent_dir,
        file_format=cfg["data"]["file_format"],
        field_names=cfg["data"]["field_names"],
        load_coords=True,                                   # <-- CA coords on
        subsample=cfg["data"].get("subsample"),
        rng=rng,
        length_range=tuple(cfg["data"]["length_range"]) if cfg["data"].get("length_range") else None,
    )
    logger.info("Loaded %d protein records", len(records))

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
    logger.info("Held-out test: %d proteins, Training: %d proteins", len(test_ids), len(train_ids))

    with open(output_dir / "heldout_test_ids.txt", "w") as f:
        for pid in test_ids:
            f.write(pid + "\n")

    fold_df = create_fold_assignments(train_ids, n_folds=5)
    fold_df.to_csv(output_dir / "fold_assignments.csv", index=False)
    logger.info("Fold assignments saved (%d proteins, 5 folds)", len(fold_df))

    record_by_id = {r.protein_id: r for r in records}

    if args.smoke_test:
        fold_indices = [0]
    elif args.folds is not None:
        fold_indices = [int(x) for x in args.folds.split(",")]
    else:
        fold_indices = list(range(5))

    all_fold_results = []
    for fold_idx in fold_indices:
        val_ids = set(fold_df[fold_df["fold"] == fold_idx]["protein_id"])
        fold_train_ids = set(fold_df[fold_df["fold"] != fold_idx]["protein_id"])

        fold_train_records = [record_by_id[pid] for pid in sorted(fold_train_ids) if pid in record_by_id]
        fold_val_records = [record_by_id[pid] for pid in sorted(val_ids) if pid in record_by_id]

        logger.info("Fold %d: %d train, %d val proteins",
                     fold_idx, len(fold_train_records), len(fold_val_records))

        result = train_fold(
            fold_idx=fold_idx,
            train_records=fold_train_records,
            val_records=fold_val_records,
            prop_df=prop_df,
            output_dir=output_dir,
            config=train_config,
            device=device,
        )
        all_fold_results.append(result)

    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv(output_dir / "results_per_fold.csv", index=False)
    logger.info("Per-fold results saved to %s", output_dir / "results_per_fold.csv")

    if len(all_fold_results) > 1:
        summary_rows = []
        for pname in PROPERTY_NAMES:
            col = f"r2_{pname}"
            vals = [r[col] for r in all_fold_results if not np.isnan(r.get(col, np.nan))]
            summary_rows.append({
                "property": pname,
                "r2_mean": np.mean(vals) if vals else np.nan,
                "r2_std": np.std(vals) if vals else np.nan,
                "n_folds": len(vals),
            })
        mean_vals = [r["r2_mean"] for r in all_fold_results if not np.isnan(r.get("r2_mean", np.nan))]
        summary_rows.append({
            "property": "MEAN",
            "r2_mean": np.mean(mean_vals) if mean_vals else np.nan,
            "r2_std": np.std(mean_vals) if mean_vals else np.nan,
            "n_folds": len(mean_vals),
        })
        pd.DataFrame(summary_rows).to_csv(output_dir / "results_summary.csv", index=False)
        logger.info("Summary results saved")

    logger.info("All outputs in %s", output_dir)
    return output_dir


if __name__ == "__main__":
    main()
