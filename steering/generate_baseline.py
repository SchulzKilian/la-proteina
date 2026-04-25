"""Generate unguided La-Proteina samples with lengths matched to a reference
distribution (e.g. the 300-800 training-set length distribution).

Use this to produce a baseline population for natural-vs-generated comparisons
on developability metrics. The model is loaded once; samples are generated in
a single loop over (seed, length) pairs drawn from the empirical distribution.

Usage:
    python -m steering.generate_baseline \\
        --proteina_config inference_ucond_notri_long \\
        --n_samples 100 \\
        --length_csv /rds/user/ks2218/hpc-work/developability_panel.csv \\
        --length_col sequence_length \\
        --output_dir results/generated_baseline_300_800 \\
        --nsteps 200
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import lightning as L  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from proteinfoundation.generate import load_ckpt_n_configure_inference  # noqa: E402
from steering.generate import (  # noqa: E402
    load_proteina_config, generate_one, save_protein,
)

logger = logging.getLogger(__name__)


def sample_matched_lengths(
    length_csv: str,
    length_col: str,
    n: int,
    length_range: tuple[int, int] | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n integer lengths from the empirical distribution in the CSV.

    Filters to length_range if given. Samples WITH replacement from the
    empirical distribution so the generated-sample length histogram matches
    the reference.
    """
    df = pd.read_csv(length_csv, usecols=[length_col])
    lens = df[length_col].dropna().astype(int).values
    if length_range is not None:
        lo, hi = length_range
        lens = lens[(lens >= lo) & (lens <= hi)]
    if len(lens) == 0:
        raise ValueError(f"No lengths found in {length_csv} after filtering.")
    return rng.choice(lens, size=n, replace=True)


def sample_stratified_lengths(
    length_range: tuple[int, int],
    n_per_bin: int,
    bin_width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stratified-uniform sampling: n_per_bin lengths drawn uniformly per bin.

    Bins partition [lo, hi) in steps of bin_width. The returned array is
    shuffled so the slow long-length samples are interleaved with short ones —
    this way, hitting the SLURM time limit costs proportionally from every bin
    instead of losing the upper bins entirely.
    """
    lo, hi = length_range
    edges = list(range(lo, hi + 1, bin_width))
    if edges[-1] != hi:
        edges.append(hi)
    parts = []
    for b_lo, b_hi in zip(edges[:-1], edges[1:]):
        parts.append(rng.integers(b_lo, b_hi, size=n_per_bin))
    lengths = np.concatenate(parts)
    rng.shuffle(lengths)
    return lengths


def main():
    parser = argparse.ArgumentParser(description="Unguided baseline generation, length-matched")
    parser.add_argument("--proteina_config", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--length_csv", type=str,
                        default="/rds/user/ks2218/hpc-work/developability_panel.csv")
    parser.add_argument("--length_col", type=str, default="sequence_length")
    parser.add_argument("--length_range", type=int, nargs=2, default=[300, 800])
    parser.add_argument("--seed_base", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--nsteps", type=int, default=None)
    parser.add_argument("--length_mode", choices=["empirical", "stratified"],
                        default="empirical",
                        help="empirical: sample lengths from --length_csv. "
                             "stratified: uniform within fixed bins of --bin_width.")
    parser.add_argument("--n_per_bin", type=int, default=100,
                        help="Used only when --length_mode=stratified. "
                             "Total samples = n_per_bin * num_bins; --n_samples is ignored.")
    parser.add_argument("--bin_width", type=int, default=50,
                        help="Bin width for --length_mode=stratified.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Sample lengths
    rng = np.random.default_rng(args.seed_base)
    if args.length_mode == "stratified":
        lengths = sample_stratified_lengths(
            length_range=tuple(args.length_range),
            n_per_bin=args.n_per_bin,
            bin_width=args.bin_width,
            rng=rng,
        )
        logger.info(
            "Stratified sampling: %d bins of width %d, %d per bin -> %d total",
            (args.length_range[1] - args.length_range[0]) // args.bin_width,
            args.bin_width, args.n_per_bin, len(lengths),
        )
    else:
        lengths = sample_matched_lengths(
            length_csv=args.length_csv,
            length_col=args.length_col,
            n=args.n_samples,
            length_range=tuple(args.length_range) if args.length_range else None,
            rng=rng,
        )
    logger.info(
        "Sampled %d lengths (mean=%.1f, std=%.1f, min=%d, max=%d)",
        len(lengths), lengths.mean(), lengths.std(), lengths.min(), lengths.max(),
    )

    # Load model
    logger.info("Loading La-Proteina config: %s", args.proteina_config)
    cfg = load_proteina_config(args.proteina_config)
    if args.nsteps is not None:
        cfg.generation.args.nsteps = args.nsteps
    logger.info("Loading model checkpoint (nsteps=%d)...", cfg.generation.args.nsteps)
    model = load_ckpt_n_configure_inference(cfg)
    model = model.to(device)
    model.eval()
    model.steering_guide = None  # explicit: no guidance

    # Output setup
    out_dir = Path(args.output_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Manifest of (seed, length) pairs
    manifest = []

    # Generate
    t0 = time.time()
    for i, length in enumerate(lengths):
        seed = args.seed_base + i
        protein_id = f"s{seed}_n{int(length)}"
        logger.info("[%d/%d] %s ...", i + 1, len(lengths), protein_id)
        try:
            coors, res_type, mask, _ = generate_one(model, int(length), int(seed), device)
            pdb_path, pt_path = save_protein(coors, res_type, mask, protein_id, samples_dir)
            manifest.append({
                "protein_id": protein_id,
                "seed": int(seed),
                "length": int(length),
                "pdb": str(pdb_path.relative_to(out_dir)),
                "pt": str(pt_path.relative_to(out_dir)),
            })
        except Exception as e:
            logger.error("Failed to generate %s: %s", protein_id, e)
            manifest.append({
                "protein_id": protein_id,
                "seed": int(seed),
                "length": int(length),
                "error": str(e),
            })

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(lengths) - i - 1) / max(rate, 1e-9)
        logger.info("  elapsed=%.1fs, ETA=%.1fs", elapsed, eta)

        # Write manifest after each sample so nothing is lost on crash
        pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)

    # Run config
    run_config = {
        "proteina_config": args.proteina_config,
        "length_mode": args.length_mode,
        "n_samples_total": int(len(lengths)),
        "length_csv": args.length_csv if args.length_mode == "empirical" else None,
        "length_col": args.length_col,
        "length_range": args.length_range,
        "n_per_bin": args.n_per_bin if args.length_mode == "stratified" else None,
        "bin_width": args.bin_width if args.length_mode == "stratified" else None,
        "seed_base": args.seed_base,
        "nsteps": cfg.generation.args.nsteps,
        "device": str(device),
        "length_stats": {
            "mean": float(lengths.mean()),
            "std": float(lengths.std()),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
        },
    }
    with open(out_dir / "run_config.yaml", "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)

    logger.info("Done. %d samples in %s", len(lengths), out_dir)


if __name__ == "__main__":
    main()
