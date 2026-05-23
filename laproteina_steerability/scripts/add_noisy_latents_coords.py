"""Fine-tune the CA-conditioned multi-task property predictor on noisy (latent, CA) pairs.

Mirrors `add_noisy_latents.py` exactly — same t-window, same Langevin term,
same hyperparameters, same fold-splits-from-src-run, same checkpoint format —
except that:

  * the source run is a `multitask_t1_coords/<ts>/` run (CA-conditioned, NOT
    the latent-only run);
  * EACH item now applies the forward interpolant to BOTH the latent and the
    CA coordinates at the *same* per-protein `t`. This matches the steering-time
    distribution where the SDE flow advances both channels together.

Noise model (per protein, per epoch draw):

    t      ~ U(t_min, t_max)
    eps_z  ~ N(0, I)  shape [L, 8]
    eps_c  ~ N(0, I)  shape [L, 3]
    z_t    = (1 - t) * eps_z + t * z_1
    c_t    = (1 - t) * eps_c + t * c_1
    if sigma_langevin > 0:
      eps_z2, eps_c2 ~ N(0, I)
      z_t += sigma_langevin * sqrt(t*(1-t)) * eps_z2
      c_t += sigma_langevin * sqrt(t*(1-t)) * eps_c2

The latent noise model is byte-identical to the existing latent-only fine-tune.
The CA noise model uses the same shape `(1-t)·eps + t·c_1` because the la-proteina
CA flow matcher uses the same forward interpolant (rdn_flow_matcher.py:91-115);
matching it during predictor training is what makes the steering-time input
in-distribution.

Outputs land under `laproteina_steerability/logs/multitask_t1_coords_noise_aware/<ts>/`,
mirroring the latent-only output layout.
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset
from src.multitask_predictor_coords.dataset_coords import (
    PROPERTY_NAMES,
    PropertyDatasetCoords,
    ZScoreStats,
    LengthBucketBatchSampler,
    collate_fn_coords,
)
from src.multitask_predictor_coords.model_coords import PropertyTransformerCoords
from src.multitask_predictor_coords.train_coords import evaluate, _arch_kwargs
from src.part2_property_probes.properties import (
    load_properties,
    align_properties_to_latents,
)

logger = logging.getLogger(__name__)


class NoisyPropertyDatasetCoords(PropertyDatasetCoords):
    """PropertyDatasetCoords that noises BOTH latents and CA at the same t.

    Per __getitem__:
      t      ~ U(t_min, t_max)
      z_t = (1-t)*eps_z + t*z_1                                  (+optional Langevin)
      c_t = (1-t)*eps_c + t*c_1                                  (+optional Langevin)
    """

    def __init__(
        self,
        records,
        prop_df,
        stats,
        t_min: float,
        t_max: float,
        sigma_langevin: float,
        seed: int = 42,
        fix_t_input: float | None = None,
    ):
        super().__init__(records, prop_df, stats=stats, t_value=1.0)
        if not (0.0 <= t_min < t_max <= 1.0):
            raise ValueError(f"need 0 <= t_min < t_max <= 1, got [{t_min}, {t_max}]")
        if sigma_langevin < 0:
            raise ValueError(f"sigma_langevin must be >= 0, got {sigma_langevin}")
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.sigma_langevin = float(sigma_langevin)
        self.fix_t_input = None if fix_t_input is None else float(fix_t_input)
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        z_1 = item["latents"]    # [L, 8]
        c_1 = item["coords"]     # [L, 3]

        t = float(self._rng.uniform(self.t_min, self.t_max))
        eps_z = torch.randn_like(z_1)
        eps_c = torch.randn_like(c_1)
        z_t = (1.0 - t) * eps_z + t * z_1
        c_t = (1.0 - t) * eps_c + t * c_1

        if self.sigma_langevin > 0.0:
            eps_z2 = torch.randn_like(z_1)
            eps_c2 = torch.randn_like(c_1)
            scale = (t * (1.0 - t)) ** 0.5
            z_t = z_t + self.sigma_langevin * scale * eps_z2
            c_t = c_t + self.sigma_langevin * scale * eps_c2

        item["latents"] = z_t
        item["coords"] = c_t
        item["t"] = t if self.fix_t_input is None else self.fix_t_input
        return item


def _worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    base = info.dataset._seed
    info.dataset._rng = np.random.default_rng(base + worker_id + 1)


def _eval_loader(model, loader, device, stats):
    return evaluate(model, loader, device, stats)


def fine_tune_fold(
    fold_idx: int,
    src_ckpt_path: Path,
    train_records,
    val_records,
    prop_df,
    output_dir: Path,
    cfg: dict,
    device: torch.device,
) -> dict:
    logger.info("=== Fold %d (fine-tune from %s) ===", fold_idx, src_ckpt_path)

    src = torch.load(src_ckpt_path, map_location="cpu", weights_only=False)
    stats = ZScoreStats(mean=src["stats_mean"], std=src["stats_std"])

    # Rebuild model with the exact arch the src ckpt was trained with.
    arch = src.get("arch_kwargs", None)
    if arch is None:
        # Fall back to constructing from cfg — works as long as cfg matches src run.
        arch = _arch_kwargs(cfg)
    model = PropertyTransformerCoords(**arch).to(device)
    model.load_state_dict(src["model_state_dict"])
    logger.info("  Loaded source weights (epoch=%d, src val_r2_mean=%.4f)",
                src.get("epoch", -1), src.get("val_r2_mean", float("nan")))

    train_ds = NoisyPropertyDatasetCoords(
        train_records, prop_df, stats=stats,
        t_min=cfg["t_min"], t_max=cfg["t_max"],
        sigma_langevin=cfg["sigma_langevin"],
        seed=42 + fold_idx * 1000,
        fix_t_input=cfg.get("fix_t_input"),
    )
    val_ds_noisy = NoisyPropertyDatasetCoords(
        val_records, prop_df, stats=stats,
        t_min=cfg["t_min"], t_max=cfg["t_max"],
        sigma_langevin=cfg["sigma_langevin"],
        seed=43 + fold_idx * 1000,
        fix_t_input=cfg.get("fix_t_input"),
    )
    val_ds_clean = PropertyDatasetCoords(val_records, prop_df, stats=stats, t_value=1.0)

    bs = cfg["batch_size"]
    train_sampler = LengthBucketBatchSampler(
        [r.length for r in train_records], batch_size=bs, shuffle=True, seed=42 + fold_idx,
    )
    val_sampler_noisy = LengthBucketBatchSampler(
        [r.length for r in val_records], batch_size=bs, shuffle=False,
    )
    val_sampler_clean = LengthBucketBatchSampler(
        [r.length for r in val_records], batch_size=bs, shuffle=False,
    )

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=collate_fn_coords,
        num_workers=4, pin_memory=True, persistent_workers=True,
        worker_init_fn=_worker_init_fn,
    )
    val_loader_noisy = DataLoader(
        val_ds_noisy, batch_sampler=val_sampler_noisy, collate_fn=collate_fn_coords,
        num_workers=2, pin_memory=True, persistent_workers=True,
        worker_init_fn=_worker_init_fn,
    )
    val_loader_clean = DataLoader(
        val_ds_clean, batch_sampler=val_sampler_clean, collate_fn=collate_fn_coords,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    total_steps = cfg["max_epochs"] * len(train_loader)
    lr_floor = cfg.get("lr_floor", 0.1) * cfg["lr"]
    floor_ratio = lr_floor / cfg["lr"]
    def _lr_lambda(step):
        progress = min(step / max(total_steps, 1), 1.0)
        return floor_ratio + (1.0 - floor_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    curves_path = output_dir / "training_curves.csv"
    write_header = not curves_path.exists()
    curves_file = open(curves_path, "a", newline="")
    fields = ["fold", "step", "total_loss", "lr"] + [f"mse_{p}" for p in PROPERTY_NAMES]
    curves_writer = csv.DictWriter(curves_file, fieldnames=fields)
    if write_header:
        curves_writer.writeheader()

    epoch_path = output_dir / "epoch_metrics.csv"
    write_eheader = not epoch_path.exists()
    epoch_file = open(epoch_path, "a", newline="")
    epoch_fields = (
        ["fold", "epoch", "train_loss",
         "r2_mean_noisy", "r2_mean_t1"]
        + [f"r2_{p}_noisy" for p in PROPERTY_NAMES]
        + [f"r2_{p}_t1"    for p in PROPERTY_NAMES]
    )
    epoch_writer = csv.DictWriter(epoch_file, fieldnames=epoch_fields)
    if write_eheader:
        epoch_writer.writeheader()

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"fold_{fold_idx}_best.pt"

    best_r2 = -float("inf")
    patience_counter = 0
    patience = cfg["patience"]
    step_counter = 0

    for epoch in range(cfg["max_epochs"]):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            latents = batch["latents"].to(device)
            coords = batch["coords"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)
            t = batch["t"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                preds = model(latents, coords, mask, t)
                valid = ~torch.isnan(targets)
                diff = (preds - targets) ** 2
                diff = diff * valid.float()
                per_prop_mse = diff.sum(dim=0) / valid.float().sum(dim=0).clamp(min=1.0)
                loss = per_prop_mse.mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step_counter += 1
            total_loss += loss.item()
            n_batches += 1

            row = {
                "fold": fold_idx, "step": step_counter,
                "total_loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
            }
            for i, pname in enumerate(PROPERTY_NAMES):
                row[f"mse_{pname}"] = per_prop_mse[i].item()
            curves_writer.writerow(row)

        train_loss = total_loss / max(n_batches, 1)
        curves_file.flush()

        val_noisy = _eval_loader(model, val_loader_noisy, device, stats)
        val_clean = _eval_loader(model, val_loader_clean, device, stats)
        elapsed = time.time() - t0

        logger.info(
            "Fold %d Epoch %d: train=%.4f r2_noisy=%.4f r2_t1=%.4f (%.1fs)",
            fold_idx, epoch, train_loss,
            val_noisy["r2_mean"], val_clean["r2_mean"], elapsed,
        )

        epoch_row = {
            "fold": fold_idx, "epoch": epoch, "train_loss": train_loss,
            "r2_mean_noisy": val_noisy["r2_mean"],
            "r2_mean_t1": val_clean["r2_mean"],
        }
        for p in PROPERTY_NAMES:
            epoch_row[f"r2_{p}_noisy"] = val_noisy.get(f"r2_{p}", float("nan"))
            epoch_row[f"r2_{p}_t1"] = val_clean.get(f"r2_{p}", float("nan"))
        epoch_writer.writerow(epoch_row)
        epoch_file.flush()

        if val_noisy["r2_mean"] > best_r2:
            best_r2 = val_noisy["r2_mean"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "arch_kwargs": arch,
                "epoch": epoch,
                "val_r2_mean": best_r2,
                "val_r2_mean_t1": val_clean["r2_mean"],
                "val_results_noisy": val_noisy,
                "val_results_t1": val_clean,
                "stats_mean": stats.mean,
                "stats_std": stats.std,
                "coord_conditioned": True,
                "noise_aware": True,
                "t_min": cfg["t_min"],
                "t_max": cfg["t_max"],
                "sigma_langevin": cfg["sigma_langevin"],
                "fix_t_input": cfg.get("fix_t_input"),
                "src_ckpt": str(src_ckpt_path),
            }, ckpt_path)
            logger.info("  New best r2_noisy=%.4f, saved %s", best_r2, ckpt_path.name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("  Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    curves_file.close()
    epoch_file.close()

    return {
        "fold": fold_idx,
        "best_r2_noisy": best_r2,
        "ckpt": str(ckpt_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-run", type=Path, required=True,
                    help="Source CA-conditioned training run dir (must contain config.yaml + checkpoints/fold_*_best.pt)")
    ap.add_argument("--folds", type=str, default="0,1,2,3,4",
                    help="Comma-separated fold indices to fine-tune")

    ap.add_argument("--t-min", type=float, default=0.3)
    ap.add_argument("--t-max", type=float, default=0.8)
    ap.add_argument("--sigma-langevin", type=float, default=0.1)
    ap.add_argument("--fix-t-input", type=float, default=None)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--out-root", type=Path,
                    default=Path("logs/multitask_t1_coords_noise_aware"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    src_run = args.src_run
    if not src_run.is_absolute():
        src_run = ROOT / src_run
    src_cfg_path = src_run / "config.yaml"
    if not src_cfg_path.exists():
        raise FileNotFoundError(f"Source config not found: {src_cfg_path}")
    src_cfg = yaml.safe_load(src_cfg_path.read_text())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    out_dir = out_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "source_run.txt").write_text(str(src_run) + "\n")

    cfg = {
        # arch (mirrored from src so the rebuilt model exactly matches)
        "latent_dim": src_cfg.get("latent_dim", 8),
        "d_model": src_cfg.get("d_model", 128),
        "n_heads": src_cfg.get("n_heads", 4),
        "n_layers": src_cfg.get("n_layers", 3),
        "ffn_expansion": src_cfg.get("ffn_expansion", 4),
        "dropout": src_cfg.get("dropout", 0.1),
        "max_len": src_cfg.get("max_len", 1024),
        "n_rbf": src_cfg.get("n_rbf", 32),
        "rbf_max_nm": src_cfg.get("rbf_max_nm", 8.0),
        "relpos_clamp": src_cfg.get("relpos_clamp", 32),
        "relpos_dim": src_cfg.get("relpos_dim", 32),
        # training
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        # noise
        "t_min": args.t_min,
        "t_max": args.t_max,
        "sigma_langevin": args.sigma_langevin,
        "fix_t_input": args.fix_t_input,
        # provenance
        "src_run": str(src_run),
        "timestamp": timestamp,
    }
    (out_dir / "config.yaml").write_text(yaml.dump(cfg, default_flow_style=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s, output dir: %s", device, out_dir)
    logger.info("Noise: t ~ U(%.2f, %.2f), sigma_langevin=%.3f, fix_t_input=%s",
                cfg["t_min"], cfg["t_max"], cfg["sigma_langevin"],
                "real_noisy_t" if cfg["fix_t_input"] is None else f"{cfg['fix_t_input']:.2f}")

    data_cfg = src_cfg["data"]
    latent_dir = data_cfg["latent_dir"]
    if not Path(latent_dir).is_dir():
        fallback = ROOT.parent / "data" / "pdb_train" / "processed_latents_300_800"
        if fallback.is_dir():
            latent_dir = str(fallback.resolve())
            logger.info("Latent dir fallback: %s", latent_dir)
        else:
            raise FileNotFoundError(f"Latent dir not found: {data_cfg['latent_dir']}")

    rng = np.random.default_rng(42)
    records = load_dataset(
        latent_dir=latent_dir,
        file_format=data_cfg["file_format"],
        field_names=data_cfg["field_names"],
        load_coords=True,
        subsample=data_cfg.get("subsample"),
        rng=rng,
        length_range=tuple(data_cfg["length_range"]) if data_cfg.get("length_range") else None,
    )
    logger.info("Loaded %d protein records (with CA coords)", len(records))

    prop_names = src_cfg["property_names"]
    default_cfg_path = ROOT / "config" / "default.yaml"
    if default_cfg_path.exists():
        default_cfg = yaml.safe_load(default_cfg_path.read_text())
        prop_gran = default_cfg.get("part2", {}).get("property_granularity", {})
        prop_gran = {p: prop_gran.get(p, "protein") for p in prop_names}
    else:
        prop_gran = {p: "protein" for p in prop_names}

    prop_df = load_properties(
        property_file=src_cfg["property_file"],
        property_names=prop_names,
        property_granularity=prop_gran,
    )
    records, prop_df = align_properties_to_latents(
        records, prop_df,
        property_names=prop_names,
        property_granularity=prop_gran,
    )
    logger.info("After alignment: %d proteins", len(records))

    fold_df = pd.read_csv(src_run / "fold_assignments.csv")
    record_by_id = {r.protein_id: r for r in records}

    fold_indices = [int(x) for x in args.folds.split(",")]

    summary = []
    for fold_idx in fold_indices:
        src_ckpt = src_run / "checkpoints" / f"fold_{fold_idx}_best.pt"
        if not src_ckpt.exists():
            logger.error("Missing source checkpoint for fold %d: %s", fold_idx, src_ckpt)
            continue

        val_ids = set(fold_df[fold_df["fold"] == fold_idx]["protein_id"])
        train_ids = set(fold_df[fold_df["fold"] != fold_idx]["protein_id"])
        train_recs = [record_by_id[p] for p in sorted(train_ids) if p in record_by_id]
        val_recs = [record_by_id[p] for p in sorted(val_ids) if p in record_by_id]
        logger.info("Fold %d: %d train, %d val", fold_idx, len(train_recs), len(val_recs))

        result = fine_tune_fold(
            fold_idx=fold_idx,
            src_ckpt_path=src_ckpt,
            train_records=train_recs,
            val_records=val_recs,
            prop_df=prop_df,
            output_dir=out_dir,
            cfg=cfg,
            device=device,
        )
        summary.append(result)

    if summary:
        pd.DataFrame(summary).to_csv(out_dir / "results_per_fold.csv", index=False)
    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
