"""Dense-t fixed-seed validation comparison of two raw (non-EMA) checkpoints
from the CA-only training run, to determine whether the late-training val-loss
uptick is signal or noise.

Loads two raw .ckpt files (NOT the -EMA companions), runs the validation set
through each at a fixed grid of t values with deterministic noise, and reports:
  - per-checkpoint mean loss across all (batch, t) pairs
  - per-t mean loss for each checkpoint (the loss-vs-t curve)
  - paired difference distribution (matched batch+t pairs)

Why raw, not EMA: the wandb val loss the user is diagnosing was logged on the
raw training model, so the EMA companion would smooth across the uptick and
defeat the purpose of the comparison.

Usage:
    python -m analysis_cheap_diagnostics.dense_t_eval
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from proteinfoundation.proteina import Proteina
from proteinfoundation.utils.fetch_last_ckpt import fetch_last_ckpt  # noqa: F401

CKPT_DIR = ROOT / "store/test_ca_only_diffusion/1776805213/checkpoints"
CKPT_A = CKPT_DIR / "best_val_00000021_000000002204.ckpt"  # the "best val" point
CKPT_B = CKPT_DIR / "best_val_00000024_000000002457.ckpt"  # the "uptick" point

CONFIG_DIR = str(ROOT / "configs")
CONFIG_NAME = "training_ca_only"

OUT = ROOT / "analysis_cheap_diagnostics"
OUT.mkdir(exist_ok=True)


def set_seed(s: int) -> None:
    """Set every RNG source the model touches to a deterministic seed."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_proteina_raw(ckpt_path: Path, device: torch.device) -> Proteina:
    """Load Proteina from a RAW checkpoint file (not the -EMA companion).

    Lightning's load_from_checkpoint reads `state_dict` from the ckpt — the
    base model weights, not EMA. The cfg_exp comes from `hyper_parameters`.
    """
    print(f"  loading {ckpt_path.name}")
    model = Proteina.load_from_checkpoint(
        str(ckpt_path), map_location=device, strict=True,
    )
    model.eval()
    model = model.to(device)
    # Force self_cond OFF for deterministic eval. The training validation runs
    # through training_step which randomly applies self-conditioning 50% of the
    # time — adding variance we don't want. Both checkpoints see identical
    # conditions this way; the relative comparison is the point.
    model.cfg_exp.training.self_cond = False
    if hasattr(model, "steering_guide"):
        model.steering_guide = None
    return model


def force_constant_t(model: Proteina, t_value: float) -> None:
    """Monkey-patch fm.sample_t to return constant t for every example.

    Returns Dict[data_mode -> tensor] matching what fm.sample_t normally returns.
    For CA-only mode, only "bb_ca" is in data_modes.
    """
    fm = model.fm

    def sample_t_const(shape, device):
        return {
            dm: torch.full(shape, t_value, device=device, dtype=torch.float32)
            for dm in fm.data_modes
        }

    fm.sample_t = sample_t_const  # type: ignore[assignment]


def cache_val_batches(model: Proteina, n_batches: int, device: torch.device) -> list[dict]:
    """Build the same datamodule training used and pull n_batches val batches."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg_exp = compose(config_name=CONFIG_NAME)
    # Mirror what train.load_data_module does: instantiate the datamodule.
    import hydra
    cfg_data = cfg_exp.dataset
    cfg_data.datamodule.num_workers = 4  # fewer workers; we only need a few batches
    datamodule = hydra.utils.instantiate(cfg_data.datamodule)
    datamodule.setup("validate")
    loader = datamodule.val_dataloader()
    print(f"  val_dataloader has {len(loader)} batches (batch_size={cfg_data.datamodule.batch_size})")

    cached = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        # Move tensors to device
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(device)
            else:
                moved[k] = v
        cached.append(moved)
    print(f"  cached {len(cached)} val batches in RAM")
    return cached


def eval_one_pass(model: Proteina, batch: dict, t_value: float, base_seed: int) -> float:
    """Run a single validation forward pass at constant t with deterministic noise.

    Mirrors the loss aggregation in proteina.training_step (sum of means of
    non-_justlog losses).
    """
    set_seed(base_seed)
    force_constant_t(model, t_value)

    with torch.no_grad():
        # 1. Add clean samples
        b = {k: v for k, v in batch.items()}  # shallow copy
        b = model.add_clean_samples(b)
        # 2. Corrupt (uses our constant-t sample_t and seeded noise)
        b = model.fm.corrupt_batch(b)
        # 3. Self-cond OFF (we set it in load); add empty x_sc handling
        b = model.handle_self_cond(b)
        # 4. Folding/inv-folding probability is 0.0 in CA-only config
        b = model.handle_folding_n_inverse_folding(b)
        # 5. Forward + loss
        nn_out = model.call_nn(b, n_recycle=0)
        losses = model.fm.compute_loss(batch=b, nn_out=nn_out)
        loss_total = sum(
            torch.mean(losses[k]).item()
            for k in losses
            if "_justlog" not in k
        )
    return float(loss_total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_batches", type=int, default=32,
                        help="Number of val batches to evaluate (each batch = 6 proteins).")
    parser.add_argument("--n_t", type=int, default=20,
                        help="Number of t values in the dense grid.")
    parser.add_argument("--t_min", type=float, default=0.05)
    parser.add_argument("--t_max", type=float, default=0.95)
    parser.add_argument("--seed_base", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoints:\n  A: {CKPT_A.name}\n  B: {CKPT_B.name}")

    t_grid = np.linspace(args.t_min, args.t_max, args.n_t)
    print(f"t-grid: {t_grid.round(3).tolist()}")

    # Load checkpoint A and pull val batches via that model's datamodule
    print("\n[Load A]")
    model_a = load_proteina_raw(CKPT_A, device)
    print("\n[Cache val batches]")
    val_batches = cache_val_batches(model_a, args.n_batches, device)
    if not val_batches:
        raise RuntimeError("No val batches cached — datamodule returned empty.")

    # Eval A
    print("\n[Eval A across t-grid]")
    rows = []
    t0 = time.time()
    for t_idx, t_val in enumerate(t_grid):
        for b_idx, batch in enumerate(val_batches):
            seed = args.seed_base + 1000 * t_idx + b_idx
            loss = eval_one_pass(model_a, batch, float(t_val), seed)
            rows.append({"ckpt": "A_step2204", "t_idx": t_idx, "t": float(t_val),
                         "batch_idx": b_idx, "seed": seed, "loss": loss})
        elapsed = time.time() - t0
        n_done = (t_idx + 1) * len(val_batches)
        n_total = args.n_t * len(val_batches)
        print(f"  t={t_val:.3f}: {len(val_batches)} batches done. "
              f"elapsed={elapsed:.1f}s, ETA={elapsed/n_done*(n_total-n_done):.1f}s")
    del model_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Eval B
    print("\n[Load B]")
    model_b = load_proteina_raw(CKPT_B, device)
    print("\n[Eval B across t-grid]")
    t0 = time.time()
    for t_idx, t_val in enumerate(t_grid):
        for b_idx, batch in enumerate(val_batches):
            seed = args.seed_base + 1000 * t_idx + b_idx
            loss = eval_one_pass(model_b, batch, float(t_val), seed)
            rows.append({"ckpt": "B_step2457", "t_idx": t_idx, "t": float(t_val),
                         "batch_idx": b_idx, "seed": seed, "loss": loss})
        elapsed = time.time() - t0
        n_done = (t_idx + 1) * len(val_batches)
        n_total = args.n_t * len(val_batches)
        print(f"  t={t_val:.3f}: {len(val_batches)} batches done. "
              f"elapsed={elapsed:.1f}s, ETA={elapsed/n_done*(n_total-n_done):.1f}s")

    # Write raw results
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dense_t_eval_raw.csv", index=False)
    print(f"\nWrote {OUT/'dense_t_eval_raw.csv'} ({len(df)} rows)")

    # Aggregate
    print("\n" + "=" * 70)
    print("HEADLINE: per-checkpoint mean loss across all (batch, t) pairs")
    print("=" * 70)
    agg = df.groupby("ckpt")["loss"].agg(["mean", "std", "count"])
    print(agg.to_string())
    diff = agg.loc["B_step2457", "mean"] - agg.loc["A_step2204", "mean"]
    print(f"\n  B - A = {diff:+.5f}  (positive means step 2457 is WORSE, i.e. uptick is real)")

    # Per-t curves
    print("\n" + "=" * 70)
    print("Per-t loss for each checkpoint (loss-vs-t curve)")
    print("=" * 70)
    per_t = df.groupby(["ckpt", "t"])["loss"].mean().unstack("ckpt")
    per_t["B_minus_A"] = per_t["B_step2457"] - per_t["A_step2204"]
    print(per_t.to_string())
    per_t.to_csv(OUT / "dense_t_eval_per_t.csv")

    # Paired (batch, t) differences — matched comparison, lower variance
    print("\n" + "=" * 70)
    print("Paired (batch, t) differences  ckpt_B - ckpt_A")
    print("=" * 70)
    a = df[df["ckpt"] == "A_step2204"].set_index(["t_idx", "batch_idx"])["loss"]
    b = df[df["ckpt"] == "B_step2457"].set_index(["t_idx", "batch_idx"])["loss"]
    paired = (b - a).dropna()
    print(f"  n_pairs = {len(paired)}")
    print(f"  mean diff = {paired.mean():+.5f}")
    print(f"  std diff  = {paired.std():.5f}")
    print(f"  SE of mean = {paired.std() / np.sqrt(len(paired)):.5f}")
    t_stat = paired.mean() / (paired.std() / np.sqrt(len(paired)))
    print(f"  t-stat (paired) = {t_stat:.3f}")
    print(f"  fraction of pairs with B>A: {(paired > 0).mean():.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
