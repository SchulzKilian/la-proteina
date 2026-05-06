"""Fine-tune the multi-task property predictor on noisy latents.

Goal: stop the steering predictor from being gradient-hacked. The original
predictor was trained on clean t=1 AE-mean latents, but at sampling time
the steering hook feeds it `z_t` from an SDE trajectory (`product space
flow matcher`, eq. 3 in `rdn_flow_matcher.py:288`). That distribution is
*not* what the predictor was trained on, so adversarial off-manifold
directions exist and the steering gradient happily walks toward them.
Result on the May-04 ensemble run: predictor claims ΔTANGO=-288 at w=16,
real binary says -34 (~8.5x over-claim).

Fix: fine-tune each fold checkpoint on z_t drawn from the same
distribution the predictor will see at sampling time, restricted to the
steering window `t ∈ [t_min, t_max]`.

Noise model (per the SDE inference setup):

    z_t = (1 - t) * eps_1 + t * z_1                  # forward interpolant
    z_t += sigma_langevin * sqrt(t * (1 - t)) * eps_2  # extra Langevin term

  where eps_1, eps_2 ~ N(0, I) and z_1 is the AE-mean latent.

Note on Langevin term (flag the user asked for):

  In flow matching, the *marginal* p_t(z_t) under SDE sampling equals the
  marginal under ODE sampling, equals the closed-form interpolant — that
  is a theorem of the stochastic-interpolant framework (assuming the
  score is exact). So strictly speaking the Langevin add-on shifts the
  training distribution slightly *off* what the SDE actually visits in
  expectation, and a principled run uses sigma_langevin=0.

  Two reasons to keep it small-but-nonzero:
    (a) the score is not exact — the SDE trajectories deviate from the
        ideal marginal, and sigma_langevin > 0 widens the training
        support enough to cover that drift;
    (b) it doubles as Tikhonov-style data augmentation, hardening the
        predictor against the high-frequency adversarial directions
        that gradient hacking exploits.

  Default sigma_langevin = 0.1 (~10% of latent std). Set to 0.0 if you
  want the principled-marginal-only training.

Outputs (originals at `logs/multitask_t1/<src>/checkpoints/` are untouched):

    laproteina_steerability/logs/multitask_t1_noise_aware/<timestamp>/
      config.yaml
      source_run.txt
      training_curves.csv
      epoch_metrics.csv
      checkpoints/fold_{0..4}_best.pt
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

logger = logging.getLogger(__name__)


class NoisyPropertyDataset(PropertyDataset):
    """PropertyDataset that draws z_t from the SDE forward interpolant + optional Langevin term.

    On every __getitem__:
      t      ~ U(t_min, t_max)
      eps_1  ~ N(0, I)  shape [L, 8]
      z_t    = (1 - t) * eps_1 + t * z_1
      if sigma_langevin > 0:
        eps_2 ~ N(0, I)
        z_t  += sigma_langevin * sqrt(t * (1 - t)) * eps_2

    The t value is per-protein, so a batch contains a spread of t — exactly
    what the FiLM time conditioning was built for.
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
    ):
        super().__init__(records, prop_df, stats=stats, t_value=1.0)
        if not (0.0 <= t_min < t_max <= 1.0):
            raise ValueError(f"need 0 <= t_min < t_max <= 1, got [{t_min}, {t_max}]")
        if sigma_langevin < 0:
            raise ValueError(f"sigma_langevin must be >= 0, got {sigma_langevin}")
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.sigma_langevin = float(sigma_langevin)
        # Per-worker rng — reseeded in worker_init_fn so workers don't share noise
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        z_1 = item["latents"]  # [L, 8] torch.float32

        t = float(self._rng.uniform(self.t_min, self.t_max))
        eps_1 = torch.randn_like(z_1)
        z_t = (1.0 - t) * eps_1 + t * z_1

        if self.sigma_langevin > 0.0:
            eps_2 = torch.randn_like(z_1)
            # sqrt(t*(1-t)) peaks at t=0.5, vanishes at the endpoints — same
            # qualitative shape as Brownian-bridge variance in an SDE bridge.
            scale = (t * (1.0 - t)) ** 0.5
            z_t = z_t + self.sigma_langevin * scale * eps_2

        item["latents"] = z_t
        item["t"] = t
        return item


def _worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    base = info.dataset._seed
    info.dataset._rng = np.random.default_rng(base + worker_id + 1)


@torch.no_grad()
def evaluate_clean(model, loader, device, stats) -> dict:
    """Evaluate on clean (t=1, no extra noise) data — sanity check that fine-tuning
    didn't destroy the original t=1 accuracy."""
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
    """Fine-tune one fold from src_ckpt_path on noisy latents."""
    logger.info("=== Fold %d (fine-tune from %s) ===", fold_idx, src_ckpt_path)

    # Reuse the source checkpoint's z-score stats so the new predictor's
    # output space matches the original — critical for the steering hook
    # which de-normalises with stats_mean / stats_std from the ckpt file.
    src = torch.load(src_ckpt_path, map_location="cpu", weights_only=False)
    stats = ZScoreStats(mean=src["stats_mean"], std=src["stats_std"])

    train_ds = NoisyPropertyDataset(
        train_records, prop_df, stats=stats,
        t_min=cfg["t_min"], t_max=cfg["t_max"],
        sigma_langevin=cfg["sigma_langevin"],
        seed=42 + fold_idx * 1000,
    )
    # Validation also uses the same noise distribution — that's the regime
    # we care about for steering. We additionally evaluate on clean t=1
    # at the end of each epoch as a sanity probe.
    val_ds_noisy = NoisyPropertyDataset(
        val_records, prop_df, stats=stats,
        t_min=cfg["t_min"], t_max=cfg["t_max"],
        sigma_langevin=cfg["sigma_langevin"],
        seed=43 + fold_idx * 1000,
    )
    val_ds_clean = PropertyDataset(val_records, prop_df, stats=stats, t_value=1.0)

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
        train_ds, batch_sampler=train_sampler, collate_fn=collate_fn,
        num_workers=4, pin_memory=True, persistent_workers=True,
        worker_init_fn=_worker_init_fn,
    )
    val_loader_noisy = DataLoader(
        val_ds_noisy, batch_sampler=val_sampler_noisy, collate_fn=collate_fn,
        num_workers=2, pin_memory=True, persistent_workers=True,
        worker_init_fn=_worker_init_fn,
    )
    val_loader_clean = DataLoader(
        val_ds_clean, batch_sampler=val_sampler_clean, collate_fn=collate_fn,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    # Build model with the same hyperparameters as the source run, then load weights.
    model = PropertyTransformer(
        latent_dim=cfg["latent_dim"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        ffn_expansion=cfg["ffn_expansion"],
        dropout=cfg["dropout"],
        n_properties=len(PROPERTY_NAMES),
        max_len=cfg["max_len"],
    ).to(device)
    model.load_state_dict(src["model_state_dict"])
    logger.info("  Loaded source weights (epoch=%d, src val_r2_mean=%.4f)",
                src.get("epoch", -1), src.get("val_r2_mean", float("nan")))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    # Cosine decay across the full epoch budget — pulls LR from cfg["lr"] down
    # to lr_floor at the last optimizer step. Helps the late epochs lock in
    # rather than oscillate; first run had several folds bouncing ±0.02 r² in
    # epochs 7-9 at constant LR=1e-4.
    total_steps = cfg["max_epochs"] * len(train_loader)
    lr_floor = cfg.get("lr_floor", 0.1) * cfg["lr"]   # default decay to 10× lower
    floor_ratio = lr_floor / cfg["lr"]
    def _lr_lambda(step):
        progress = min(step / max(total_steps, 1), 1.0)
        return floor_ratio + (1.0 - floor_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Curves CSV (per-step)
    curves_path = output_dir / "training_curves.csv"
    write_header = not curves_path.exists()
    curves_file = open(curves_path, "a", newline="")
    fields = ["fold", "step", "total_loss", "lr"] + [f"mse_{p}" for p in PROPERTY_NAMES]
    curves_writer = csv.DictWriter(curves_file, fieldnames=fields)
    if write_header:
        curves_writer.writeheader()

    # Epoch CSV
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
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)
            t = batch["t"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                preds = model(latents, mask, t)
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

        val_noisy = evaluate(model, val_loader_noisy, device, stats)
        val_clean = evaluate_clean(model, val_loader_clean, device, stats)
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

        # Early stop on the noisy R² — that's the regime steering uses.
        if val_noisy["r2_mean"] > best_r2:
            best_r2 = val_noisy["r2_mean"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_r2_mean": best_r2,
                "val_r2_mean_t1": val_clean["r2_mean"],
                "val_results_noisy": val_noisy,
                "val_results_t1": val_clean,
                "stats_mean": stats.mean,
                "stats_std": stats.std,
                "noise_aware": True,
                "t_min": cfg["t_min"],
                "t_max": cfg["t_max"],
                "sigma_langevin": cfg["sigma_langevin"],
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
    ap.add_argument("--src-run", type=Path,
                    default=Path("logs/multitask_t1/20260427_161809"),
                    help="Source training run dir (must contain config.yaml + checkpoints/fold_*_best.pt)")
    ap.add_argument("--folds", type=str, default="0,1,2,3,4",
                    help="Comma-separated fold indices to fine-tune")

    # Noise model
    ap.add_argument("--t-min", type=float, default=0.3,
                    help="Min t for the steering window (matches schedule.t_start in steering configs)")
    ap.add_argument("--t-max", type=float, default=0.8,
                    help="Max t for the steering window (matches schedule.t_end)")
    ap.add_argument("--sigma-langevin", type=float, default=0.1,
                    help="Extra Langevin noise scale; 0.0 = principled marginal only (forward interpolant)")

    # Training (fine-tune defaults: shorter, lower LR than the from-scratch run)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    # Plumbing
    ap.add_argument("--out-root", type=Path,
                    default=Path("logs/multitask_t1_noise_aware"))
    ap.add_argument("--data-config", type=Path, default=None,
                    help="Optional path to a YAML with the same data:/part2: blocks as the steerability default config. Defaults to the src-run's config.yaml.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Resolve source run
    src_run = args.src_run
    if not src_run.is_absolute():
        src_run = ROOT / src_run
    src_cfg_path = src_run / "config.yaml"
    if not src_cfg_path.exists():
        raise FileNotFoundError(f"Source config not found: {src_cfg_path}")
    src_cfg = yaml.safe_load(src_cfg_path.read_text())

    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    out_dir = out_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "source_run.txt").write_text(str(src_run) + "\n")

    # Build the run config (architecture inherited from src; training hyperparams from CLI)
    cfg = {
        # arch
        "latent_dim": src_cfg.get("latent_dim", 8),
        "d_model": src_cfg.get("d_model", 128),
        "n_heads": src_cfg.get("n_heads", 4),
        "n_layers": src_cfg.get("n_layers", 3),
        "ffn_expansion": src_cfg.get("ffn_expansion", 4),
        "dropout": src_cfg.get("dropout", 0.1),
        "max_len": src_cfg.get("max_len", 1024),
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
        # provenance
        "src_run": str(src_run),
        "timestamp": timestamp,
    }
    (out_dir / "config.yaml").write_text(yaml.dump(cfg, default_flow_style=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s, output dir: %s", device, out_dir)
    logger.info("Noise: t ~ U(%.2f, %.2f), sigma_langevin=%.3f",
                cfg["t_min"], cfg["t_max"], cfg["sigma_langevin"])

    # Load data — use the data config from the src run, with the same fallback logic
    # the original run.py uses (primary path, then la-proteina/data/pdb_train symlink).
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
        load_coords=False,
        subsample=data_cfg.get("subsample"),
        rng=rng,
        length_range=tuple(data_cfg["length_range"]) if data_cfg.get("length_range") else None,
    )
    logger.info("Loaded %d protein records", len(records))

    # Properties — pulled from src config. The saved config.yaml flattens away the
    # part2.property_granularity dict, so reconstruct it from the part2 block of the
    # steerability default config (or fall back to all-"protein" — the entire trained
    # set is per-protein aggregates from the developability panel).
    prop_names = src_cfg["property_names"]
    default_cfg_path = ROOT / "config" / "default.yaml"
    if default_cfg_path.exists():
        default_cfg = yaml.safe_load(default_cfg_path.read_text())
        prop_gran = default_cfg.get("part2", {}).get("property_granularity", {})
        # Restrict to properties present in src run, fill missing with "protein".
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

    # Reuse the source run's exact fold + held-out splits — otherwise ensemble
    # mixing across folds at sampling time would silently leak.
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
