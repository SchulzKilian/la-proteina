"""Per-t validation loss for any CA-only Proteina checkpoint, bypassing the
broken PDBLightningDataModule (which insists on downloading metadata from
the public internet).

Builds its own dataloader from `data/pdb_train/processed_latents/<shard>/*.pt`
files on disk (each is a PyG `Data` object with `coords_nm` [N, 37, 3],
`coord_mask` [N, 37], `residue_type` [N]). Picks a deterministic subset,
applies the standard training transforms (center + global rotation +
chain-break feature), then for each t-bucket in
{[0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1.0]} samples t uniformly
from that bucket per protein and computes the FM loss.

Output: JSON file with per-bucket mean loss + sample count per bucket.

Usage:
    python proteinfoundation/run_per_t_val.py \
        --ckpt_name best_val_00000026_000000002646.ckpt \
        --label canonical_2646 \
        --num_proteins 600
"""
import os
import sys
import argparse
import glob
import json
import random
from typing import Dict, List

root = os.path.abspath(".")
sys.path.insert(0, root)

import lightning as L
import torch
from omegaconf import OmegaConf
from dotenv import load_dotenv
from loguru import logger
from torch_geometric.data import Data, Batch as PyGBatch

from proteinfoundation.proteina import Proteina
from proteinfoundation.datasets.transforms import (
    CoordsToNanometers,
    CenterStructureTransform,
    GlobalRotationTransform,
    ChainBreakPerResidueTransform,
)

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")


def list_processed_files(data_dir: str) -> List[str]:
    """Return all .pt files under processed_latents/<shard>/, sorted for determinism."""
    pattern = os.path.join(data_dir, "pdb_train", "processed_latents", "*", "*.pt")
    files = sorted(glob.glob(pattern))
    return files


def pick_subset(files: List[str], n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    return files[:n]


def load_one(path: str) -> Data:
    return torch.load(path, map_location="cpu", weights_only=False)


def pad_and_collate(items: List[Data], max_pad: int) -> Dict[str, torch.Tensor]:
    """Pad each protein to length max_pad and stack into dense tensors.
    Returns dict with keys coords_nm, coord_mask, residue_type, mask."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_name", required=True)
    parser.add_argument("--ckpt_path", default="/home/ks2218/la-proteina")
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--num_proteins",
        type=int,
        default=600,
        help="proteins per bucket (sampled with replacement so each bucket is balanced).",
    )
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--max_pad", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results/per_t_val")
    parser.add_argument("--data_dir", default=os.environ.get("DATA_PATH", "/home/ks2218/la-proteina/data"))
    args = parser.parse_args()

    load_dotenv()
    L.seed_everything(args.seed)

    ckpt_file = os.path.join(args.ckpt_path, args.ckpt_name)
    assert os.path.exists(ckpt_file), f"Missing ckpt: {ckpt_file}"

    logger.info(f"Loading {ckpt_file}")
    model = Proteina.load_from_checkpoint(
        ckpt_file, strict=False, autoencoder_ckpt_path=None
    )
    cfg_exp = model.cfg_exp
    run_name = cfg_exp.get("run_name_")
    logger.info(f"  run_name_   = {run_name}")
    assert "local_latents" not in cfg_exp.get("product_flowmatcher", {}), (
        "This script is CA-only; ckpt has local_latents. Use the AE ckpt path."
    )

    device = torch.device("cuda")
    model.to(device).eval()

    # Build the protein set (deterministic subset of processed_latents/).
    all_files = list_processed_files(args.data_dir)
    if len(all_files) < args.num_proteins:
        logger.warning(
            f"Only {len(all_files)} files available, requested {args.num_proteins}"
        )
    files = pick_subset(all_files, args.num_proteins, seed=args.seed)
    logger.info(f"  using {len(files)} proteins from {args.data_dir}/pdb_train/processed_latents/")

    # Filter by length (skip > max_pad; matches training's max_length=512 cutoff).
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
    logger.info(f"  after length filter (≤ {args.max_pad}): {len(items)} proteins")

    # Apply training-style transforms (with a fixed seed for reproducibility across ckpts).
    # CoordsToNanometers is a no-op since our files already have coords_nm in nm.
    # CenterStructureTransform: subtract center of mass.
    # GlobalRotationTransform: apply a random rotation. Use a per-protein RNG seeded
    # from a fixed master so the SAME proteins get the SAME rotations across all 3 ckpts.
    transforms = [
        CenterStructureTransform(),
        ChainBreakPerResidueTransform(),
    ]
    rotation = GlobalRotationTransform()
    # Apply non-rotation transforms first
    for i, it in enumerate(items):
        for t in transforms:
            it = t(it)
        # rotation: seed with (master + i) so each protein has its own rotation
        # but the rotation IS the same for the same protein across ckpts.
        torch.manual_seed(args.seed + 1_000 + i)
        items[i] = rotation(it)

    # Run per-bucket evaluation.
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    bucket_results = {}

    for lo, hi in buckets:
        bucket_key = f"t_{int(lo*100):03d}_{int(hi*100):03d}"
        logger.info(f"  bucket [{lo:.1f}, {hi:.1f}) ...")
        per_protein_losses = []
        # Iterate batches
        for bs_start in range(0, len(items), args.batch_size):
            bs_items = items[bs_start : bs_start + args.batch_size]
            batch_cpu = pad_and_collate(bs_items, args.max_pad)
            batch = {k: v.to(device) for k, v in batch_cpu.items()}

            # Build x_1 via the model's own getter (returns coords_nm[:, :, 1, :] = CA)
            batch = model.add_clean_samples(batch)

            # Manual corrupt — controlled t in [lo, hi]
            x_1_dict, mask_proc, batch_shape, n_pad, dtype, dev = (
                model.fm.process_batch(batch)
            )
            x_0 = model.fm.sample_noise(
                n=n_pad, shape=batch_shape, mask=mask_proc, device=dev
            )
            B = batch_shape[0]
            t_bb = torch.empty(B, device=dev).uniform_(lo, hi)
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

        n_used = len(per_protein_losses)
        mean_loss = sum(per_protein_losses) / max(1, n_used)
        std_loss = (
            (sum((x - mean_loss) ** 2 for x in per_protein_losses) / max(1, n_used)) ** 0.5
        )
        sem = std_loss / max(1, n_used) ** 0.5
        bucket_results[bucket_key] = {
            "mean": mean_loss,
            "std": std_loss,
            "sem": sem,
            "n": n_used,
            "t_lo": lo,
            "t_hi": hi,
        }
        logger.info(
            f"    bucket [{lo:.1f}, {hi:.1f}): n={n_used} mean={mean_loss:.4f} ±{sem:.4f} (sem)"
        )

    # Print summary
    print()
    print(f"=== Per-t validation loss for {args.label} ===")
    print(f"  ckpt        : {ckpt_file}")
    print(f"  run_name_   : {run_name}")
    print(f"  num_proteins (after length filter): {len(items)}")
    print(f"  seed        : {args.seed}")
    print()
    print(f"  {'bucket':<14} {'mean':>10} {'sem':>10} {'std':>10} {'n':>6}")
    for k, v in bucket_results.items():
        print(f"  {k:<14} {v['mean']:>10.4f} {v['sem']:>10.4f} {v['std']:>10.4f} {v['n']:>6d}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.label}.json")
    payload = {
        "label": args.label,
        "ckpt_path": ckpt_file,
        "run_name_": run_name,
        "num_proteins_after_filter": len(items),
        "seed": args.seed,
        "max_pad": args.max_pad,
        "buckets": bucket_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  → wrote {out_path}")


if __name__ == "__main__":
    main()
