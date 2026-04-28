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
import os
import re
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
    parser.add_argument("--run_until_timeout", action="store_true",
                        help="Ignore --n_samples / --n_per_bin and keep generating "
                             "until --time_budget_s elapses (or SLURM end-time minus "
                             "--slurm_safety_s, whichever is sooner).")
    parser.add_argument("--time_budget_s", type=float, default=None,
                        help="Wall-clock budget for the generation loop (seconds). "
                             "Only used with --run_until_timeout.")
    parser.add_argument("--slurm_safety_s", type=float, default=120.0,
                        help="If $SLURM_JOB_END_TIME is set, exit this many seconds "
                             "before it to leave room for cleanup. Default 120s.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Output setup (must happen before we count existing samples for resume)
    out_dir = Path(args.output_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Resume: scan existing samples for used seeds, so re-runs against the same
    # output_dir append rather than collide. Filenames are s{seed}_n{length}.pt.
    existing_pts = sorted(samples_dir.glob("s*_n*.pt"))
    seed_re = re.compile(r"^s(\d+)_n\d+$")
    used_seeds: set[int] = set()
    for p in existing_pts:
        m = seed_re.match(p.stem)
        if m:
            used_seeds.add(int(m.group(1)))
    next_seed = max(used_seeds) + 1 if used_seeds else args.seed_base
    if used_seeds:
        logger.info(
            "Resume: %d existing samples in %s (seeds %d..%d). Generating more, "
            "starting at seed %d.",
            len(existing_pts), samples_dir, min(used_seeds), max(used_seeds),
            next_seed,
        )
    else:
        logger.info("No existing samples in %s; starting fresh at seed %d.",
                    samples_dir, next_seed)

    # Load existing manifest so we append rather than overwrite
    manifest_path = out_dir / "manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path).to_dict("records")
    else:
        manifest = []

    # Decide loop mode
    rng = np.random.default_rng(next_seed)  # length-sampling RNG independent of seed counter
    n_bins = (args.length_range[1] - args.length_range[0]) // args.bin_width
    bin_edges = list(range(args.length_range[0], args.length_range[1] + 1, args.bin_width))
    if bin_edges[-1] != args.length_range[1]:
        bin_edges.append(args.length_range[1])

    def draw_length(step_count: int) -> int:
        """One length per call. Stratified: round-robin over bins. Empirical:
        sample with replacement from the CSV column."""
        if args.length_mode == "stratified":
            b = step_count % n_bins
            return int(rng.integers(bin_edges[b], bin_edges[b + 1]))
        else:
            # Lazy load + cache empirical lengths in a closure attribute
            cache = getattr(draw_length, "_emp", None)
            if cache is None:
                df = pd.read_csv(args.length_csv, usecols=[args.length_col])
                lens = df[args.length_col].dropna().astype(int).values
                lo, hi = args.length_range
                lens = lens[(lens >= lo) & (lens <= hi)]
                if len(lens) == 0:
                    raise ValueError(f"No lengths found in {args.length_csv} after filtering.")
                draw_length._emp = lens  # type: ignore[attr-defined]
                cache = lens
            return int(rng.choice(cache))

    # Time budget for run-until-timeout mode
    deadline: float | None = None
    if args.run_until_timeout:
        candidates: list[float] = []
        if args.time_budget_s is not None:
            candidates.append(time.time() + args.time_budget_s)
        slurm_end = os.environ.get("SLURM_JOB_END_TIME")
        if slurm_end:
            candidates.append(float(slurm_end) - args.slurm_safety_s)
        if not candidates:
            raise SystemExit(
                "--run_until_timeout requires either --time_budget_s or "
                "$SLURM_JOB_END_TIME to be set."
            )
        deadline = min(candidates)
        logger.info(
            "Run-until-timeout mode: deadline in %.1f min (%.0f s).",
            (deadline - time.time()) / 60.0, deadline - time.time(),
        )
    else:
        # Compute fixed total for the bounded mode
        if args.length_mode == "stratified":
            total = n_bins * args.n_per_bin
        else:
            total = args.n_samples
        logger.info("Bounded mode: will generate %d new samples.", total)

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

    # Generate
    t0 = time.time()
    seed = next_seed
    step = 0  # round-robin counter for stratified mode
    new_count = 0
    while True:
        # Termination
        if deadline is not None:
            if time.time() >= deadline:
                logger.info("Deadline reached; stopping cleanly.")
                break
        else:
            if new_count >= total:
                break

        length = draw_length(step)
        protein_id = f"s{seed}_n{int(length)}"
        if deadline is not None:
            remaining = deadline - time.time()
            logger.info("[+%d new | seed=%d, len=%d] %s ... (%.0f s left)",
                        new_count, seed, length, protein_id, remaining)
        else:
            logger.info("[%d/%d] %s ...", new_count + 1, total, protein_id)

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
        rate = (new_count + 1) / elapsed
        logger.info("  elapsed=%.1fs, rate=%.2f/min", elapsed, rate * 60.0)

        # Atomic-ish manifest write so SLURM kill mid-write can't truncate it.
        tmp = manifest_path.with_suffix(".csv.tmp")
        pd.DataFrame(manifest).to_csv(tmp, index=False)
        os.replace(tmp, manifest_path)

        seed += 1
        step += 1
        new_count += 1

    # Run config (overwrite each run; reflects the most recent invocation)
    run_config = {
        "proteina_config": args.proteina_config,
        "length_mode": args.length_mode,
        "run_until_timeout": bool(args.run_until_timeout),
        "n_samples_total_in_manifest": len(manifest),
        "n_new_this_run": new_count,
        "length_csv": args.length_csv if args.length_mode == "empirical" else None,
        "length_col": args.length_col,
        "length_range": args.length_range,
        "n_per_bin": args.n_per_bin if args.length_mode == "stratified" else None,
        "bin_width": args.bin_width if args.length_mode == "stratified" else None,
        "seed_base": args.seed_base,
        "nsteps": cfg.generation.args.nsteps,
        "device": str(device),
    }
    with open(out_dir / "run_config.yaml", "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)

    logger.info("Done. %d new samples this run; %d total in %s",
                new_count, len(manifest), out_dir)


if __name__ == "__main__":
    main()
