"""Aggregate validation loss under the training-time t-distribution
(`mix_unif_beta(1.9, 1.0, 0.02)`) for two CA-only checkpoints, repeated
across N independent t-draw seeds. Used to test whether the wandb-logged
`validation_loss/loss_epoch` flip between canonical and variants is
within the noise floor of this estimator (which is dominated by the
1/(1-t)^2 weight in `rdn_flow_matcher.compute_fm_loss`).

Same paired 600-protein subset as `run_per_t_val.py` (seed=42 for the
protein subset and per-protein rotation). Per (ckpt, t_seed): one
aggregate number = mean over the 600 proteins of the FM loss with t
drawn once per protein from `mix_unif_beta(1.9, 1.0, 0.02)`.

Output: results/aggregate_val_seeds/<run_label>.json with
{"per_ckpt": {ckpt_label: [agg_seed_0, agg_seed_1, ...]}}.

Usage:
    python proteinfoundation/run_aggregate_val_seeds.py \
        --ckpts best_val_00000026_000000002646.ckpt:canonical_2646 \
                best_val_00000012_000000001259.ckpt:sparse_vanilla_1259 \
        --num_proteins 600 --n_seeds 20
"""
import os
import sys
import argparse
import glob
import json
import random
from typing import Dict, List, Tuple

root = os.path.abspath(".")
sys.path.insert(0, root)

import lightning as L
import torch
from dotenv import load_dotenv
from loguru import logger
from torch_geometric.data import Data

from proteinfoundation.proteina import Proteina
from proteinfoundation.datasets.transforms import (
    CenterStructureTransform,
    GlobalRotationTransform,
    ChainBreakPerResidueTransform,
)

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")


def list_processed_files(data_dir: str) -> List[str]:
    pattern = os.path.join(data_dir, "pdb_train", "processed_latents", "*", "*.pt")
    return sorted(glob.glob(pattern))


def pick_subset(files: List[str], n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    return files[:n]


def load_one(path: str) -> Data:
    return torch.load(path, map_location="cpu", weights_only=False)


def pad_and_collate(items: List[Data], max_pad: int) -> Dict[str, torch.Tensor]:
    bs = len(items)
    coords = torch.zeros(bs, max_pad, 37, 3)
    cmask = torch.zeros(bs, max_pad, 37, dtype=torch.bool)
    rtype = torch.zeros(bs, max_pad, dtype=torch.long)
    mask = torch.zeros(bs, max_pad, dtype=torch.bool)
    for i, it in enumerate(items):
        n = it.coords_nm.shape[0]
        if n > max_pad:
            n = max_pad
        coords[i, :n] = it.coords_nm[:n]
        cmask[i, :n] = it.coord_mask[:n]
        rtype[i, :n] = it.residue_type[:n]
        mask[i, :n] = True
    return {
        "coords_nm": coords,
        "coord_mask": cmask,
        "residue_type": rtype,
        "mask": mask,
    }


def sample_t_mix_unif_beta(
    shape: Tuple[int, ...], p1: float, p2: float, p3: float,
    generator: torch.Generator, device: torch.device,
) -> torch.Tensor:
    """Same logic as `_sample_t(name='mix_unif_beta')` but seeded via an
    explicit `torch.Generator` so we can reproduce a draw exactly."""
    # `torch.distributions.beta.Beta.sample` does not accept a generator,
    # so we synthesize a Beta(p1, p2) draw via the Gamma trick:
    #     X ~ Gamma(p1, 1), Y ~ Gamma(p2, 1) → X/(X+Y) ~ Beta(p1, p2)
    g_x = torch._standard_gamma(
        torch.full(shape, p1, device=device),
    ) if False else None  # unused; kept for documentation
    # Use rejection: easier to call torch.distributions with manual_seed.
    # We re-seed the global RNG for this op only, then restore it.
    state = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
    seed_int = int(generator.initial_seed())
    if device.type == "cuda":
        torch.cuda.manual_seed(seed_int)
    else:
        torch.manual_seed(seed_int)
    samples_beta = torch.distributions.beta.Beta(p1, p2).sample(shape).to(device)
    samples_uniform = torch.rand(shape, device=device, generator=generator)
    u = torch.rand(shape, device=device, generator=generator)
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    else:
        torch.set_rng_state(state)
    return torch.where(u < p3, samples_uniform, samples_beta)


def build_protein_subset(args, device) -> List[Data]:
    L.seed_everything(args.subset_seed)
    all_files = list_processed_files(args.data_dir)
    files = pick_subset(all_files, args.num_proteins, seed=args.subset_seed)
    items: List[Data] = []
    for fpath in files:
        try:
            d = load_one(fpath)
        except Exception as e:
            logger.warning(f"skip {fpath}: {e}")
            continue
        if d.coords_nm.shape[0] > args.max_pad:
            continue
        items.append(d)
    transforms = [CenterStructureTransform(), ChainBreakPerResidueTransform()]
    rotation = GlobalRotationTransform()
    for i, it in enumerate(items):
        for t_fn in transforms:
            it = t_fn(it)
        torch.manual_seed(args.subset_seed + 1_000 + i)
        items[i] = rotation(it)
    logger.info(f"  built protein subset: {len(items)} proteins (after length filter ≤ {args.max_pad})")
    return items


def aggregate_one_seed(
    model: Proteina,
    items: List[Data],
    args,
    device: torch.device,
    t_seed: int,
) -> Tuple[float, int, float, float]:
    """One full-pass over the 600-protein subset with one fresh t-draw per
    protein from `mix_unif_beta(p1, p2, p3)`. Returns
    (mean, n, std, max) of the per-protein FM loss."""
    L.seed_everything(t_seed)
    per_protein_losses: List[float] = []
    for bs_start in range(0, len(items), args.batch_size):
        bs_items = items[bs_start : bs_start + args.batch_size]
        batch_cpu = pad_and_collate(bs_items, args.max_pad)
        batch = {k: v.to(device) for k, v in batch_cpu.items()}
        batch = model.add_clean_samples(batch)
        x_1_dict, mask_proc, batch_shape, n_pad, dtype, dev = (
            model.fm.process_batch(batch)
        )
        x_0 = model.fm.sample_noise(
            n=n_pad, shape=batch_shape, mask=mask_proc, device=dev
        )
        B = batch_shape[0]
        # Fresh per-protein t-draw from mix_unif_beta(p1, p2, p3).
        beta_dist = torch.distributions.beta.Beta(args.beta_p1, args.beta_p2)
        samples_beta = beta_dist.sample((B,)).to(dev)
        samples_uniform = torch.rand(B, device=dev)
        u = torch.rand(B, device=dev)
        t_bb = torch.where(u < args.beta_p3, samples_uniform, samples_beta)
        t = {"bb_ca": t_bb}
        x_t = model.fm.interpolate(x_0=x_0, x_1=x_1_dict, t=t, mask=mask_proc)
        batch["x_0"] = x_0
        batch["x_1"] = x_1_dict
        batch["x_t"] = x_t
        batch["t"] = t
        batch["mask"] = mask_proc
        with torch.no_grad():
            nn_out = model.call_nn(batch, n_recycle=0)
            losses = model.fm.compute_loss(batch=batch, nn_out=nn_out)
        per_proto = sum(losses[k] for k in losses if "_justlog" not in k)  # [B]
        per_protein_losses.extend(per_proto.detach().cpu().tolist())
    n = len(per_protein_losses)
    mean = sum(per_protein_losses) / max(1, n)
    var = sum((x - mean) ** 2 for x in per_protein_losses) / max(1, n)
    std = var ** 0.5
    mx = max(per_protein_losses) if per_protein_losses else float("nan")
    return mean, n, std, mx


def parse_ckpt_arg(s: str) -> Tuple[str, str]:
    """`<ckpt_name>:<label>`."""
    if ":" not in s:
        raise ValueError(f"--ckpts must use <name>:<label>, got {s}")
    name, label = s.split(":", 1)
    return name, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpts",
        nargs="+",
        required=True,
        help="One or more <ckpt_name>:<label> pairs. Both files live in --ckpt_path.",
    )
    parser.add_argument("--ckpt_path", default="/home/ks2218/la-proteina")
    parser.add_argument("--num_proteins", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--max_pad", type=int, default=512)
    parser.add_argument("--subset_seed", type=int, default=42)
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--seed_offset", type=int, default=10_000)
    parser.add_argument("--beta_p1", type=float, default=1.9)
    parser.add_argument("--beta_p2", type=float, default=1.0)
    parser.add_argument("--beta_p3", type=float, default=0.02)
    parser.add_argument(
        "--out",
        default="results/aggregate_val_seeds/agg_canonical_vs_sparse_vanilla.json",
    )
    parser.add_argument(
        "--data_dir", default=os.environ.get("DATA_PATH", "/home/ks2218/la-proteina/data")
    )
    args = parser.parse_args()

    load_dotenv()

    device = torch.device("cuda")
    items = None  # built once

    payload: Dict[str, Dict] = {
        "ckpts": [],
        "args": {
            "num_proteins": args.num_proteins,
            "max_pad": args.max_pad,
            "subset_seed": args.subset_seed,
            "n_seeds": args.n_seeds,
            "seed_offset": args.seed_offset,
            "beta_p1": args.beta_p1,
            "beta_p2": args.beta_p2,
            "beta_p3": args.beta_p3,
        },
        "per_ckpt": {},
    }

    for ckpt_arg in args.ckpts:
        ckpt_name, label = parse_ckpt_arg(ckpt_arg)
        ckpt_file = os.path.join(args.ckpt_path, ckpt_name)
        assert os.path.exists(ckpt_file), f"Missing ckpt: {ckpt_file}"
        logger.info(f"Loading {ckpt_file} (label={label})")
        model = Proteina.load_from_checkpoint(
            ckpt_file, strict=False, autoencoder_ckpt_path=None
        )
        cfg_exp = model.cfg_exp
        run_name = cfg_exp.get("run_name_")
        logger.info(f"  run_name_ = {run_name}")
        assert "local_latents" not in cfg_exp.get("product_flowmatcher", {}), (
            "CA-only only; ckpt has local_latents."
        )
        model.to(device).eval()

        if items is None:
            items = build_protein_subset(args, device)

        per_seed = []
        for k in range(args.n_seeds):
            t_seed = args.seed_offset + k
            mean, n, std, mx = aggregate_one_seed(
                model=model, items=items, args=args, device=device, t_seed=t_seed,
            )
            per_seed.append({"t_seed": t_seed, "mean": mean, "std": std, "max": mx, "n": n})
            logger.info(
                f"  [{label}] seed {k+1:02d}/{args.n_seeds} (t_seed={t_seed}): "
                f"mean={mean:.4f} std={std:.4f} max={mx:.2f} n={n}"
            )
        # Per-ckpt summary across seeds
        means = [r["mean"] for r in per_seed]
        agg_mean = sum(means) / max(1, len(means))
        agg_std = (sum((m - agg_mean) ** 2 for m in means) / max(1, len(means))) ** 0.5
        agg_min = min(means)
        agg_max = max(means)
        payload["ckpts"].append(label)
        payload["per_ckpt"][label] = {
            "ckpt_path": ckpt_file,
            "run_name_": run_name,
            "per_seed": per_seed,
            "across_seeds": {
                "mean_of_means": agg_mean,
                "std_of_means": agg_std,
                "min_of_means": agg_min,
                "max_of_means": agg_max,
                "n_seeds": len(means),
            },
        }
        logger.info(
            f"  [{label}] across {len(means)} seeds: "
            f"mean_of_means={agg_mean:.4f} std_of_means={agg_std:.4f} "
            f"min={agg_min:.4f} max={agg_max:.4f}"
        )
        # Free GPU memory before loading the next ckpt
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  → wrote {args.out}")

    # Print human-readable side-by-side
    print()
    print("=== Aggregate val loss across t-draw seeds ===")
    print(f"  protein subset: {args.num_proteins} (subset_seed={args.subset_seed})")
    print(f"  t-distribution: mix_unif_beta(p1={args.beta_p1}, p2={args.beta_p2}, p3={args.beta_p3})")
    print(f"  n seeds       : {args.n_seeds}")
    print()
    print(f"  {'label':<28} {'mean_of_means':>14} {'std_of_means':>14} {'min':>10} {'max':>10}")
    for label in payload["ckpts"]:
        s = payload["per_ckpt"][label]["across_seeds"]
        print(
            f"  {label:<28} {s['mean_of_means']:>14.4f} {s['std_of_means']:>14.4f} "
            f"{s['min_of_means']:>10.4f} {s['max_of_means']:>10.4f}"
        )


if __name__ == "__main__":
    main()
