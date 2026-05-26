"""Sidecar coords extraction for Probe 1 (pair-distance features).

Reuses the EXACT same recipe as extract_router_teacher_data.py (same seeded
protein order, same per-(protein, t) noise seed, same transforms) so the
sidecar coords reproduce the x_t that the teacher attention was captured at.

Per file out:
  <pdb_id>_t<t:.2f>_coords.pt:
    {
      "protein_id": str,
      "t_value": float,
      "N": int,
      "bb_ca": Tensor[N, 3] fp32   # x_t CA coords in nm
    }

Doesn't run any forward pass through the trunk — just the noise sampler and
interpolate. ~10× faster than the main extraction. Resumes on existing files.
"""
import argparse
import glob
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO = "/home/ks2218/la-proteina"
sys.path.insert(0, REPO)

import lightning as L
import torch
from dotenv import load_dotenv
from loguru import logger
from torch_geometric.data import Data

from proteinfoundation.datasets.transforms import (
    CenterStructureTransform,
    ChainBreakPerResidueTransform,
    GlobalRotationTransform,
)
from proteinfoundation.proteina import Proteina


CKPT_DEFAULT = (
    "/rds/user/ks2218/hpc-work/store/test_ca_only_diffusion/1776805213/"
    "checkpoints/best_val_00000026_000000002646.ckpt"
)
OUT_DIR_DEFAULT = "/rds/user/ks2218/hpc-work/store/router_teacher_data"


def list_processed_files(data_dir: str) -> List[str]:
    pattern = os.path.join(data_dir, "pdb_train", "processed_latents", "*", "*.pt")
    return sorted(glob.glob(pattern))


def load_csv_lengths(repo_root: str) -> Dict[str, int]:
    import pandas as pd
    cand = sorted(glob.glob(os.path.join(repo_root, "data", "pdb_train", "df_pdb_*.csv")))
    cand = [c for c in cand if "_latents" not in c]
    if not cand:
        cand = sorted(glob.glob(os.path.join(repo_root, "data", "pdb_train", "df_pdb_*.csv")))
    if not cand:
        return {}
    df = pd.read_csv(cand[0])
    if "id" in df.columns and "length" in df.columns:
        return dict(zip(df["id"].astype(str), df["length"].astype(int)))
    return {}


def stratified_pick(files, lengths_by_id, targets, n_per_bucket, tol, seed):
    rng = random.Random(seed)
    pool = list(files); rng.shuffle(pool)
    buckets = {L: [] for L in targets}
    need = n_per_bucket * len(targets)
    chosen = 0
    for fpath in pool:
        if chosen >= need: break
        pdb_id = os.path.splitext(os.path.basename(fpath))[0]
        n = lengths_by_id.get(pdb_id)
        if n is None:
            try:
                d = torch.load(fpath, map_location="cpu", weights_only=False)
                n = int(d.coords_nm.shape[0])
            except Exception:
                continue
        for L in targets:
            if abs(n - L) <= tol and len(buckets[L]) < n_per_bucket:
                buckets[L].append((pdb_id, fpath))
                chosen += 1
                break
    flat = []
    for L in targets:
        flat.extend(buckets[L])
        logger.info(f"  bucket L~{L}: {len(buckets[L])} / {n_per_bucket}")
    return flat


def apply_transforms(d: Data, seed: int) -> Data:
    d = CenterStructureTransform()(d)
    d = ChainBreakPerResidueTransform()(d)
    torch.manual_seed(seed)
    d = GlobalRotationTransform()(d)
    return d


def extract_coords_one(
    model,
    data_item: Data,
    pdb_id: str,
    t_values: Tuple[float, ...],
    out_dir: str,
    device,
    max_pad: int,
    seed: int,
) -> int:
    N_real = int(data_item.coords_nm.shape[0])
    n_pad = max(N_real, 64)
    if max_pad is not None:
        n_pad = min(max_pad, n_pad)
        N_real = min(N_real, n_pad)

    coords_nm = torch.zeros(1, n_pad, 37, 3)
    coord_mask = torch.zeros(1, n_pad, 37, dtype=torch.bool)
    residue_type = torch.zeros(1, n_pad, dtype=torch.long)
    mask = torch.zeros(1, n_pad, dtype=torch.bool)
    coords_nm[0, :N_real] = data_item.coords_nm[:N_real]
    coord_mask[0, :N_real] = data_item.coord_mask[:N_real]
    residue_type[0, :N_real] = data_item.residue_type[:N_real]
    mask[0, :N_real] = True
    batch_cpu = {
        "coords_nm": coords_nm,
        "coord_mask": coord_mask,
        "residue_type": residue_type,
        "mask": mask,
    }

    saved = 0
    for t_val in t_values:
        out_path = os.path.join(out_dir, f"{pdb_id}_t{t_val:.2f}_coords.pt")
        if os.path.exists(out_path):
            saved += 1
            continue
        batch = {k: v.to(device) for k, v in batch_cpu.items()}
        # MUST match extract_router_teacher_data.py's seeding to reproduce x_t.
        torch.manual_seed(seed + abs(hash((pdb_id, round(float(t_val), 4)))) % (2 ** 32))

        batch = model.add_clean_samples(batch)
        x_1_dict, mask_proc, batch_shape, n_actual, dtype, dev = model.fm.process_batch(batch)
        x_0 = model.fm.sample_noise(n=n_actual, shape=batch_shape, mask=mask_proc, device=dev)
        B = batch_shape[0]
        t_bb = torch.full((B,), float(t_val), device=dev)
        t = {"bb_ca": t_bb}
        x_t = model.fm.interpolate(x_0=x_0, x_1=x_1_dict, t=t, mask=mask_proc)
        bb_ca = x_t["bb_ca"][0, :N_real, :].detach().to(torch.float32).cpu()  # [N_real, 3] nm

        record = {
            "protein_id": pdb_id,
            "t_value": float(t_val),
            "N": int(N_real),
            "bb_ca": bb_ca,
        }
        tmp = out_path + ".tmp"
        torch.save(record, tmp)
        os.replace(tmp, out_path)
        saved += 1
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_file", default=CKPT_DEFAULT)
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--n_proteins", type=int, default=500)
    ap.add_argument("--length_targets", default="300,400,500")
    ap.add_argument("--length_tol", type=int, default=25)
    ap.add_argument("--t_values", default="0.10,0.30,0.50,0.70,0.90")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_pad", type=int, default=520)
    ap.add_argument("--data_dir", default=os.environ.get("DATA_PATH", os.path.join(REPO, "data")))
    args = ap.parse_args()

    load_dotenv()
    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    targets = tuple(int(x) for x in args.length_targets.split(","))
    t_values = tuple(float(x) for x in args.t_values.split(","))
    n_per_bucket = max(1, args.n_proteins // len(targets))

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info(f"Sidecar coords extraction → {args.out_dir}")
    logger.info(f"  targets={targets} (±{args.length_tol})  n_per_bucket={n_per_bucket}")
    logger.info(f"  t_values={t_values}  seed={args.seed}  max_pad={args.max_pad}")

    model = Proteina.load_from_checkpoint(
        args.ckpt_file, strict=False, autoencoder_ckpt_path=None
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if hasattr(model.nn, "_orig_mod"):
        model.nn = model.nn._orig_mod
    logger.info(f"  model on {device}")

    all_files = list_processed_files(args.data_dir)
    lengths_by_id = load_csv_lengths(REPO)
    selected = stratified_pick(
        all_files, lengths_by_id, targets, n_per_bucket, args.length_tol, args.seed
    )
    logger.info(f"  selected {len(selected)} proteins (mirrors main extraction order)")

    t_start = time.time()
    total_saved = 0
    for prot_idx, (pdb_id, fpath) in enumerate(selected):
        try:
            d = torch.load(fpath, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning(f"  load failed for {fpath}: {e}")
            continue
        d = apply_transforms(d, args.seed + prot_idx)
        N_real = int(d.coords_nm.shape[0])
        n_saved = extract_coords_one(
            model=model, data_item=d, pdb_id=pdb_id,
            t_values=t_values, out_dir=args.out_dir,
            device=device, max_pad=args.max_pad, seed=args.seed,
        )
        total_saved += n_saved
        if (prot_idx + 1) % 25 == 0 or prot_idx == 0:
            elapsed = time.time() - t_start
            logger.info(f"  [{prot_idx+1}/{len(selected)}] {pdb_id} N={N_real} "
                        f"saved={total_saved} ({elapsed:.1f}s)")
    elapsed = time.time() - t_start
    logger.info(f"Done. {total_saved} coords files in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
